import cvxpy as cp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import math
import os
from datetime import datetime
from src.strategic_classification.utils.gain_and_cost_func import score, f, g, f_derivative
from src.strategic_classification.utils.data_utils import split_data, load_credit_default_data, split_data
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

XDIM = 11
COST = 1/XDIM
TRAIN_SLOPE = 2
EVAL_SLOPE = 5
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

# # CCP classes

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

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c(self.x, self.r)
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
    
    def __init__(self, x_dim, funcs):
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
        
        
    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]

class BURDEN():
    
    def __init__(self, x_dim, funcs):
        self.c = funcs["c"]
        self.score = funcs["score"]

        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))

        target = self.c(self.x, self.r)
        constraints = [self.score(self.x, self.w, self.b) >= 0,
                       self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]

        objective = cp.Minimize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b], variables=[self.x])
        
        
    def calc_burden(self, X, Y, w, b):
        """
        tensor to tensor
        """
        Xpos = X[Y==1]
        if len(Xpos) == 0:
            return 0
        Xmin = self.layer(Xpos, w, b)[0]
        return torch.mean(torch.sum((Xpos-Xmin)**2, dim=1))

# # Gain & Cost functions
def c(x, r):
    return COST*cp.sum_squares(x-r)

funcs = {"f": f, "g": g, "f_derivative": f_derivative, "c": c, "score": score}

# # Data generation

X, Y = load_credit_default_data()
X, Y = X[:3000], Y[:3000]

assert(len(X[0]) == XDIM)
X, Y, Xval, Yval = split_data(X, Y, 0.5)
Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)

print("percent of positive samples: {}%".format(100 * len(Y[Y == 1]) / len(Y)))

# # Model

class MyStrategicModel(torch.nn.Module):
    def __init__(self, x_dim, funcs, train_slope, eval_slope, strategic=False, lamb=0):
        torch.manual_seed(0)
        np.random.seed(0)
        super(MyStrategicModel, self).__init__()
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))
        self.strategic = strategic
        self.lamb = lamb
        self.ccp = CCP(x_dim, funcs)
        self.delta = DELTA(x_dim, funcs)
        self.burden = BURDEN(x_dim, funcs)

    def forward(self, X, evaluation=False):
        slope = self.eval_slope if evaluation else self.train_slope
        
        if self.strategic:
            XT = self.ccp.optimize_X(X, self.w, self.b, slope)
            F_DER = self.get_f_ders(XT, slope)
            X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER) # Xopt should equal to XT but we do it again for the gradients

            output = self.score(X_opt)
        else:
            output = self.score(X)        
        return output
    
    def optimize_X(self, X, evaluation=False):
        slope = self.eval_slope if evaluation else self.train_slope
        return self.ccp.optimize_X(X, self.w, self.b, slope)
    
    def score(self, x):
        return x@self.w + self.b
    
    def get_f_ders(self, XT, slope):
        return torch.stack([0.5*slope*((slope*self.score(xt) + 1)/torch.sqrt((slope*self.score(xt) + 1)**2 + 1))*self.w for xt in XT])

    def evaluate(self, X, Y):
        Y_pred = torch.sign(self.forward(X, evaluation=True))
        num = len(Y)
        temp = Y - Y_pred
        acc = len(temp[temp == 0])*1./num        
        return acc
    
    def calc_burden(self, X, Y):
        return self.burden.calc_burden(X, Y, self.w, self.b)
    
    def loss(self, X, Y, Y_pred):
        if self.lamb > 0:
            return torch.mean(torch.clamp(1 - Y_pred * Y, min=0)) + self.lamb*self.calc_burden(X, Y)
        else:
            return torch.mean(torch.clamp(1 - Y_pred * Y, min=0))
        
    def save_model(self, X, Y, Xval, Yval, Xtest, Ytest, train_errors, val_errors, train_losses, val_losses, val_burdens, info, path, comment=None):
        if comment is not None:
            path += "_____" + comment
            
        filename = path + "/model.pt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)
        
        pd.DataFrame(X.numpy()).to_csv(path + '/X.csv')
        pd.DataFrame(Y.numpy()).to_csv(path + '/Y.csv')
        pd.DataFrame(Xval.numpy()).to_csv(path + '/Xval.csv')
        pd.DataFrame(Yval.numpy()).to_csv(path + '/Yval.csv')
        pd.DataFrame(Xval.numpy()).to_csv(path + '/Xtest.csv')
        pd.DataFrame(Yval.numpy()).to_csv(path + '/Ytest.csv')
        
        pd.DataFrame(np.array(train_errors)).to_csv(path + '/train_errors.csv')
        pd.DataFrame(np.array(val_errors)).to_csv(path + '/val_errors.csv')
        pd.DataFrame(np.array(train_losses)).to_csv(path + '/train_losses.csv')
        pd.DataFrame(np.array(val_losses)).to_csv(path + '/val_losses.csv')
        pd.DataFrame(np.array(val_burdens)).to_csv(path + '/val_burdens.csv')
        
        with open(path + "/info.txt", "w") as f:
            f.write(info)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
    
    def fit(self, X, Y, Xval, Yval, Xtest, Ytest, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None, calc_train_errors=False, comment=None):
        train_dset = TensorDataset(X, Y)
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        opt = opt(self.parameters(), **opt_kwargs)

        train_losses = []
        val_losses = []
        train_errors = []
        val_errors = []
        val_burdens = []
        
        best_val_error = 1
        consecutive_no_improvement = 0
        now = datetime.now()
        path = "./models/burden/" + now.strftime("%d-%m-%Y_%H-%M-%S")

        total_time = time.time()
        for epoch in range(epochs):
            t1 = time.time()
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in train_loader:
                opt.zero_grad()
                Ybatch_pred = self.forward(Xbatch)
                l = self.loss(Xbatch, Ybatch, Ybatch_pred)
                l.backward()
                opt.step()
                train_losses[-1].append(l.item())
                if calc_train_errors:
                    with torch.no_grad():
                        e = self.evaluate(Xbatch, Ybatch)
                        train_errors[-1].append(1-e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f" %
                          (batch, len(train_loader), np.mean(train_losses[-1])))
                batch += 1
                if callback is not None:
                    callback()

            with torch.no_grad():
                Yval_pred = self.forward(Xval, evaluation=True)
                val_loss = self.loss(Xval, Yval, Yval_pred).item()
                val_losses.append(val_loss)
                val_error = 1-self.evaluate(Xval, Yval)
                val_errors.append(val_error)
                val_burden = self.calc_burden(Xval, Yval).item()
                val_burdens.append(val_burden)
                if val_error < best_val_error:
                    consecutive_no_improvement = 0
                    best_val_error = val_error
                    if self.strategic:
                        info = "training time in seconds: {}\nepoch: {}\nbatch size: {}\ntrain slope: {}\neval slope: {}\nlearning rate: {}\nvalidation loss: {}\nvalidation error: {}\nburden: {}".format(
                        time.time()-total_time, epoch, batch_size, self.train_slope, self.eval_slope, opt_kwargs["lr"], val_loss, val_error, val_burden)
                        self.save_model(X, Y, Xval, Yval, Xtest, Ytest, train_errors, val_errors, train_losses, val_losses, val_burdens, info, path, comment)
                        print("model saved!")
                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 5:
                        break
                
            t2 = time.time()
            if verbose:
                print("----- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))
        print("training time: {} seconds".format(time.time()-total_time)) 
        return train_errors, val_errors, train_losses, val_losses

# # Train

EPOCHS = 10
BATCH_SIZE = 64

x_dim = XDIM

# non-strategic classification
print("---------- training non-strategically----------")
non_strategic_model = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, strategic=False)

fit_res_non_strategic = non_strategic_model.fit(X, Y, Xval, Yval, Xtest, Ytest,
                                opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-2)},
                                batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, calc_train_errors=False)

lambda_range = torch.logspace(start=-2, end=0.1, steps=30)
print(lambda_range)
for lamb in lambda_range:

    # strategic classification
    print("---------- training strategically----------")
    print("lambda: ", lamb.item())
    strategic_model = MyStrategicModel(x_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, lamb=lamb)

    fit_res_strategic = strategic_model.fit(X, Y, Xval, Yval, Xtest, Ytest,
                                    opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-2)},
                                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, calc_train_errors=False,
                                    comment="burden_" + str(lamb.item()))
