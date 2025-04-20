import cvxpy as cp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics import confusion_matrix
from scipy.io import arff
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
import math
import os
from src.strategic_classification.utils.gain_and_cost_func import score, f, g, f_derivative
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

TRAIN_SLOPE = 1
EVAL_SLOPE = 5
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10
SEED = 0

# Utils

def split_data(X, Y, percentage):
    num_val = int(len(X)*percentage)
    return X[num_val:], Y[num_val:], X[:num_val], Y[:num_val]

def shuffle(X, Y):
    torch.manual_seed(0)
    np.random.seed(0)
    data = torch.cat((Y, X), 1)
    data = data[torch.randperm(data.size()[0])]
    X = data[:, 1:]
    Y = data[:, 0]
    return X, Y

def conf_mat(Y1, Y2):
    num_of_samples = len(Y1)
    mat = confusion_matrix(Y1, Y2, labels=[-1, 1])*100/num_of_samples
    acc = np.trace(mat)
    return mat, acc

def calc_accuracy(Y, Ypred):
    num = len(Y)
    temp = Y - Ypred
    acc = len(temp[temp == 0])*1./num
    return acc

# Dataset

def load_spam_data():
    torch.manual_seed(0)
    np.random.seed(0)
    path = r".\tip_spam_data\IS_journal_tip_spam.arff"
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    most_disc = ['qTips_plc', 'rating_plc', 'qEmail_tip', 'qContacts_tip', 'qURL_tip', 'qPhone_tip', 'qNumeriChar_tip', 'sentistrength_tip', 'combined_tip', 'qWords_tip', 'followers_followees_gph', 'qunigram_avg_tip', 'qTips_usr', 'indeg_gph', 'qCapitalChar_tip', 'class1']
    df = df[most_disc]
    df["class1"].replace({b'spam': -1, b'notspam': 1}, inplace=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    Y = df['class1'].values
    X = df.drop('class1', axis = 1).values
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    return torch.from_numpy(X), torch.from_numpy(Y)

# CCP classes

class CCP:
    def __init__(self, x_dim, funcs, v, eps):
        self.f_derivative = funcs["f_derivative"]
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c(self.x, self.r, v, eps)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(target), constraints)
        
    def ccp(self, r):
        """
        numpy to numpy
        """
        self.xt.value = r
        self.r.value = r
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.001 and cnt < 10:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value
    
    def optimize_X(self, X, w, b, slope):
        """
        tensor to tensor
        """
        w = w.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)
        X = X.numpy()
        
        self.w.value = w
        self.b.value = b
        self.slope.value = slope
        
        return torch.stack([torch.from_numpy(self.ccp(x)) for x in X])

class DELTA():
    
    def __init__(self, x_dim, funcs, v, eps):
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r, v, eps)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
        
    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]

# Gain & Cost functions
def c(x, r, v, eps):
    return (eps*cp.sum_squares(x-r) + (1-eps)*cp.pos((x-r)@v))

# Model

class MyStrategicModel(torch.nn.Module):
    def __init__(self, x_dim, funcs, train_slope, eval_slope, v_true, eps_train, eps_true, strategic=False):
        torch.manual_seed(0)
        np.random.seed(0)

        super(MyStrategicModel, self).__init__()
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.v_true = v_true
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(1, dtype=torch.float64, requires_grad=True)))
        self.strategic = strategic
        self.eps_train = eps_train
        self.eps_true = eps_true
        self.ccp = CCP(x_dim, funcs, self.v_true, eps_train)
        self.ccp_true = CCP(x_dim, funcs, self.v_true, eps_true)
        self.delta = DELTA(x_dim, funcs, self.v_true, eps_train)

    def forward(self, X, evaluation=False):
        if self.strategic:
            if evaluation:
                XT = self.ccp.optimize_X(X, self.w, self.b, self.eval_slope)
                X_opt = XT
            else:
                XT = self.ccp.optimize_X(X, self.w, self.b, self.train_slope)
                F_DER = self.get_f_ders(XT, self.train_slope)
                X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER) # Xopt should be equal to XT but we do it again for the gradients
            output = self.score(X_opt)
        else:
            output = self.score(X)        
        return output
    
    def optimize_X(self, X):
        return self.ccp_true.optimize_X(X, self.w, self.b, self.eval_slope)
    
    def normalize_weights(self):
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(self.w**2) + self.b**2)
            self.w /= norm
            self.b /= norm

    def score(self, x):
        return x@self.w + self.b
    
    def get_f_ders(self, XT, slope):
        return torch.stack([0.5*slope*((slope*self.score(xt) + 1)/torch.sqrt((slope*self.score(xt) + 1)**2 + 1))*self.w for xt in XT])

    def calc_accuracy(self, Y, Y_pred):
        Y_pred = torch.sign(Y_pred)
        num = len(Y)
        temp = Y - Y_pred
        acc = len(temp[temp == 0])*1./num        
        return acc
    
    def evaluate(self, X, Y):      
        return self.calc_accuracy(Y, self.forward(X, evaluation=True))
    
    def loss(self, Y, Y_pred):
        return torch.mean(torch.clamp(1 - Y_pred * Y, min=0))
    
    def save_model(self, train_errors, val_errors, train_losses, val_losses, info, path, comment=None):
        if comment is not None:
            path += "/" + comment
            
        filename = path + "/model.pt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)
        
        pd.DataFrame(self.v_true).to_csv(path + '/v_true.csv')
        pd.DataFrame(np.array(train_errors)).to_csv(path + '/train_errors.csv')
        pd.DataFrame(np.array(val_errors)).to_csv(path + '/val_errors.csv')
        pd.DataFrame(np.array(train_losses)).to_csv(path + '/train_losses.csv')
        pd.DataFrame(np.array(val_losses)).to_csv(path + '/val_losses.csv')
        
        with open(path + "/info.txt", "w") as f:
            f.write(info)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
    
    def fit(self, path, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None, comment=None):
        train_dset = TensorDataset(X, Y)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        opt = opt(self.parameters(), **opt_kwargs)

        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []
        
        best_val_error = 1
        consecutive_no_improvement = 0

        total_time = time.time()
        for epoch in range(epochs):
            t1 = time.time()
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in train_loader:
                try:
                    opt.zero_grad()
                    Ybatch_pred = self.forward(Xbatch)
                    l = self.loss(Ybatch, Ybatch_pred)
                    l.backward()
                    opt.step()
                    train_losses[-1].append(l.item())
                    with torch.no_grad():
                        e = self.calc_accuracy(Ybatch, Ybatch_pred)
                        train_errors[-1].append(1-e)
                    if verbose:
                        print("batch %03d / %03d | loss: %3.5f | err: %3.5f" %
                              (batch, len(train_loader), np.mean(train_losses[-1]), np.mean(train_errors[-1])))
                    batch += 1
                    if callback is not None:
                        callback()
                except:
                    print("failed")
                
            with torch.no_grad():
                try:
                    Yval_pred = self.forward(Xval, evaluation=True)
                    val_loss = self.loss(Yval, Yval_pred).item()
                    val_losses.append(val_loss)
                    val_error = 1-self.calc_accuracy(Yval, Yval_pred)
                    val_errors.append(val_error)
                    if val_error < best_val_error:
                        consecutive_no_improvement = 0
                        best_val_error = val_error
                        info = "training time in seconds: {}\nepoch: {}\nbatch size: {}\ntrain slope: {}\neval slope: {}\nlearning rate: {}\nvalidation loss: {}\nvalidation error: {}\n".format(
                        time.time()-total_time, epoch, batch_size, self.train_slope, self.eval_slope, opt_kwargs["lr"], val_loss, val_error)
                        self.save_model(train_errors, val_errors, train_losses, val_losses, info, path, comment)
                        print("model saved!")

                    else:
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= 4:
                            break
                except:
                    print("failed")
                    
            t2 = time.time()
            if verbose:
                print("------------- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))
        print("training time: {} seconds".format(time.time()-total_time)) 
        return train_errors, val_errors, train_losses, val_losses

# Data generation

PATH = "./models/vanilla_vs_hardt_final_" + str(SEED)

path = PATH

# spam dataset
X, Y = load_spam_data()
X, Y, Xval, Yval = split_data(X, Y, 0.4)
Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)

# Train

# training parameters
x_dim = len(X[0])
epochs = 7
batch_size = 128

small_eps = 0.02
v_true = np.array([-1,-1,-1,-1,-1,-1,-1,1,1,0.1,1,0.1,0.1,1,0.1])
funcs = {"f": f, "g": g, "f_derivative": f_derivative, "c": c, "score": score}

# strategic classification
print("---------- training strategically----------")
strategic_model_approx = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, small_eps, small_eps, strategic=True)
strategic_model_approx.fit(PATH, X, Y, Xval, Yval,
                    opt=torch.optim.Adam, opt_kwargs={"lr":2*(1e-1)},
                    batch_size=batch_size, epochs=epochs, verbose=True, 
                   comment="strategic_approx")

strategic_model_approx = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, small_eps, small_eps, strategic=True)
strategic_model_approx.load_model(PATH + "/strategic_approx/model.pt")


epsilons = [0.02, 0.2, 0.4, 0.6, 0.8, 1]
for eps in epsilons:
    path = PATH + "/" + str(eps)
    print("------------------------- {} -------------------------".format(eps))
    
    # non-strategic classification
    print("---------- training non-strategically----------")
    non_strategic_model = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, eps, eps, strategic=False)
    non_strategic_model.fit(path, X, Y, Xval, Yval,
                            opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-1)},
                            batch_size=batch_size, epochs=epochs, verbose=False,
                           comment="non_strategic")

    non_strategic_model = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, eps, eps, strategic=False)
    non_strategic_model.load_model(path + "/non_strategic/model.pt")
    non_strategic_model.normalize_weights()
    
    # strategic classification
    print("---------- training strategically----------")
    strategic_model_real = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, eps, eps, strategic=True)
    strategic_model_real.fit(path, X, Y, Xval, Yval,
                        opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-1)},
                        batch_size=batch_size, epochs=epochs, verbose=False, 
                       comment="strategic_real") 

    strategic_model_real = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, eps, eps, strategic=True)
    strategic_model_real.load_model(path + "/strategic_real/model.pt")

    strategic_model_approx = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, small_eps, eps, strategic=True)
    strategic_model_approx.load_model(PATH + "/strategic_approx/model.pt")
    
    # calculate results
    accuracies = np.zeros(4)
    # non strategic model & non strategic data
    accuracies[0] = (non_strategic_model.evaluate(Xtest, Ytest))
    # approx strategic model & strategic data
    Xtest_opt = strategic_model_approx.optimize_X(Xtest)
    test_scores = strategic_model_approx.score(Xtest_opt)
    accuracies[1] = (strategic_model_approx.calc_accuracy(Ytest, test_scores))
    # real strategic model & strategic data
    Xtest_opt = strategic_model_real.optimize_X(Xtest)
    test_scores = strategic_model_real.score(Xtest_opt)
    accuracies[2] = (strategic_model_real.calc_accuracy(Ytest, test_scores))
    # non strategic model & strategic data
    Xtest_opt = non_strategic_model.optimize_X(Xtest)
    accuracies[3] = (non_strategic_model.evaluate(Xtest_opt, Ytest))

    pd.DataFrame(accuracies).to_csv(path + '/results.csv')
