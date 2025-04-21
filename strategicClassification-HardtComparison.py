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
from datetime import datetime
from src.strategic_classification.utils.gain_and_cost_func import score, f, g, f_derivative
from src.strategic_classification.utils.data_utils import split_data, load_spam_dataset
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

XDIM = 15
TRAIN_SLOPE = 1
EVAL_SLOPE = 5
GAMING = 1
EPSILON = 0.3
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10


# CCP classes

class CCP:
    def __init__(self, x_dim, funcs):
        self.f_derivative = funcs["f_derivative"]
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)
        self.v = cp.Parameter(x_dim)

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c(self.x, self.r, self.v)
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
        while diff > 0.0001 and cnt < 10:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value
    
    def optimize_X(self, X, w, b, slope, v):
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
        self.v.value = v
        
        return torch.stack([torch.from_numpy(self.ccp(x)) for x in X])

class DELTA():
    
    def __init__(self, x_dim, funcs, v):
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r, v)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
        
    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]

# Gain & Cost functions
def c_true(x, r, v):
    print(GAMING, EPSILON)
    return 2*(1./GAMING)*(EPSILON*cp.sum_squares(x-r) + (1-EPSILON)*cp.pos((x-r)@v))

def c(x, r, v):
    print(GAMING)
    return 2*(1./GAMING)*(0.01*cp.sum_squares(x-r) + (0.99)*cp.pos((x-r)@v))

funcs = {"f": f, "g": g, "f_derivative": f_derivative, "c": c, "score": score}
funcs_val = {"f": f, "g": g, "f_derivative": f_derivative, "c": c_true, "score": score}

# Data generation

X, Y = load_spam_dataset()

assert(len(X[0]) == XDIM)
X, Y, Xval, Yval = split_data(X, Y, 0.4)
Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)

print("percent of positive samples: {}%".format(100 * len(Y[Y == 1]) / len(Y)))

# Model

class MyStrategicModel(torch.nn.Module):
    def __init__(self, x_dim, funcs, funcs_val, train_slope, eval_slope, v_true, strategic=False):
        torch.manual_seed(0)
        np.random.seed(0)

        super(MyStrategicModel, self).__init__()
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.v_true = v_true
        self.v_train = v_true + 0.5*np.random.randn(x_dim)
        self.v_train -= (np.mean(self.v_train) - np.mean(v_true))
        self.v_train /= np.std(self.v_train)
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))
        self.strategic = strategic
        self.ccp = CCP(x_dim, funcs)
        self.ccp_val = CCP(x_dim, funcs_val)
        self.delta = DELTA(x_dim, funcs, self.v_train)

    def forward(self, X, evaluation=False):
        if self.strategic:
            if evaluation:
                XT = self.ccp_val.optimize_X(X, self.w, self.b, self.eval_slope, self.v_true)
                X_opt = XT
            else:
                XT = self.ccp.optimize_X(X, self.w, self.b, self.train_slope, self.v_train)
                F_DER = self.get_f_ders(XT, self.train_slope)
                X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER) # Xopt should equal to XT but we do it again for the gradients
            
            output = self.score(X_opt)
        else:
            output = self.score(X)        
        return output
    
    def optimize_X(self, X, evaluation=False):
        slope = self.eval_slope if evaluation else self.train_slope
        v = self.v_true if evaluation else self.v_train
        ccp = self.ccp_val if evaluation else self.ccp
        return ccp.optimize_X(X, self.w, self.b, slope, v)
    
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
            path += "_____" + comment
            
        filename = path + "/model.pt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)
        
        pd.DataFrame(self.v_train).to_csv(path + '/v_train.csv')
        
        pd.DataFrame(np.array(train_errors)).to_csv(path + '/train_errors.csv')
        pd.DataFrame(np.array(val_errors)).to_csv(path + '/val_errors.csv')
        pd.DataFrame(np.array(train_losses)).to_csv(path + '/train_losses.csv')
        pd.DataFrame(np.array(val_losses)).to_csv(path + '/val_losses.csv')
        
        with open(path + "/info.txt", "w") as f:
            f.write(info)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
    
    def fit(self, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None, calc_train_errors=False, comment=None):
        train_dset = TensorDataset(X, Y)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        opt = opt(self.parameters(), **opt_kwargs)

        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []
        
        best_val_error = 1
        consecutive_no_improvement = 0
        now = datetime.now()
        path = "./models/hardt/" + now.strftime("%d-%m-%Y_%H-%M-%S")

        total_time = time.time()
        for epoch in range(epochs):
            t1 = time.time()
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in train_loader:
#                 try:
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
#                 except:
#                     print("Failed")

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
                        if self.strategic:
                            info = "training time in seconds: {}\nepoch: {}\nbatch size: {}\ntrain slope: {}\neval slope: {}\nlearning rate: {}\nvalidation loss: {}\nvalidation error: {}\n".format(
                            time.time()-total_time, epoch, batch_size, self.train_slope, self.eval_slope, opt_kwargs["lr"], val_loss, val_error)
                            self.save_model(train_errors, val_errors, train_losses, val_losses, info, path, comment)
                            print("model saved!")
                
                    else:
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= 4:
                            break
                except:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 4:
                        break
                        
            t2 = time.time()
            if verbose:
                print("----- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))
        print("training time: {} seconds".format(time.time()-total_time)) 
        return train_errors, val_errors, train_losses, val_losses

# Train

EPOCHS = 10
BATCH_SIZE = 128

x_dim = XDIM
v_true = np.array([-1,-1,-1,-1,-1,-1,-1,1,1,0.1,1,0.1,0.1,1,0.1])


# non-strategic classification
print("---------- training non-strategically----------")
non_strategic_model = MyStrategicModel(x_dim, funcs, funcs_val, TRAIN_SLOPE, EVAL_SLOPE, v_true, strategic=False)

fit_res_non_strategic = non_strategic_model.fit(X, Y, Xval, Yval,
                                opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-2)},
                                batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, calc_train_errors=False)


gaming_list = np.arange(0, 3.1, 3/10)
gaming_list[0] = 0.1
print(gaming_list)
for t in gaming_list:
    print("training on t:{}".format(t))
    GAMING = t

    strategic_model = MyStrategicModel(x_dim, funcs, funcs_val, TRAIN_SLOPE, EVAL_SLOPE, v_true, strategic=True)
    fit_res_strategic = strategic_model.fit(X, Y, Xval, Yval,
                        opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-2)},
                        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, calc_train_errors=False,
                        comment="app_cost/hardt_gaming_" + str(t))
    
    strategic_model = MyStrategicModel(x_dim, funcs, funcs, TRAIN_SLOPE, EVAL_SLOPE, v_true, strategic=True)
    fit_res_strategic = strategic_model.fit(X, Y, Xval, Yval,
                        opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-2)},
                        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, calc_train_errors=False,
                        comment="real_cost/hardt_gaming_" + str(t))
