# %matplotlib notebook
import cvxpy as cp
import dccp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.metrics import zero_one_loss, confusion_matrix
from scipy.io import arff
import pandas as pd
import time
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import matplotlib.patches as mpatches
import json
import random
import math
import os, psutil
from datetime import datetime

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

XDIM = 82
TRAIN_SLOPE = 2
EVAL_SLOPE = 5
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

# # Utils

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

# # Dataset

def load_financial_distress_data(seq_len=14):
    assert(1 <= seq_len <= 14)
    torch.manual_seed(0)
    np.random.seed(0)

    data = pd.read_csv("./financial_distress_data/Financial Distress.csv")

    data = data[data.columns.drop(list(data.filter(regex='x80')))] # Since it is a categorical feature with 37 features.
    x_dim = len(data.columns) - 3
    max_seq_len = data['Time'].max()
    
    X = []
    Y = []
    data_grouped = data.groupby(['Company']).last()
    normalized_data = (data-data.mean())/data.std()
    for company_num in data_grouped.index:
        x = torch.tensor(normalized_data[data['Company'] == company_num].values)
        x = x[:,3:]
        x_seq_len = x.size()[0]
        if x_seq_len < max_seq_len:
            pad = torch.zeros((max_seq_len-x_seq_len, x_dim))
            x = torch.cat((pad, x), 0)
        y = data_grouped.iloc[company_num-1, 1]
        y = -1 if y < -0.5 else 1
        X.append(x[14-seq_len:, :])
        Y.append(y)

    XY = list(zip(X, Y))
    tmp = [list(t) for t in zip(*XY)]
    X = torch.stack(tmp[0])
    Y = torch.tensor(tmp[1])
    return X, Y

# # CCP classes

class CCP:
    def __init__(self, x_dim, h_dim, funcs):
        self.f_derivative = funcs["f_derivative"]
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.h = cp.Parameter(h_dim)
        self.w_hy = cp.Parameter(h_dim)
        self.w_hh = cp.Parameter((h_dim, h_dim))
        self.w_xh = cp.Parameter((h_dim, x_dim))
        self.slope = cp.Parameter(1)
        self.b = cp.Parameter(h_dim)

        target = self.x@self.f_derivative(self.xt, self.h, self.w_hy, self.w_hh, self.w_xh, self.b, self.slope)-self.g(self.x, self.h, self.w_hy, self.w_hh, self.w_xh, self.b, self.slope)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(target), constraints)
        
    def ccp(self, r, h):
        """
        numpy to numpy
        """
        self.xt.value = r
        self.r.value = r
        self.h.value = h
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.0001 and cnt < 100:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value
    
    def optimize_X(self, X, H, w_hy, w_hh, w_xh, b, slope):
        """
        tensor to tensor
        """
        w_hy = w_hy.detach().numpy()
        w_hh = w_hh.detach().numpy()
        w_xh = w_xh.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)
        X = X.numpy()
        H = H.detach().numpy()
        
        self.w_hy.value = w_hy
        self.w_hh.value = w_hh
        self.w_xh.value = w_xh
        self.b.value = b
        self.slope.value = slope
        
        return torch.stack([torch.from_numpy(self.ccp(x, h)) for x, h in zip(X, H)])

class DELTA():
    
    def __init__(self, x_dim, h_dim, funcs):
        self.g = funcs["g_dpp_form"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.h__w_hh_hy = cp.Parameter(1, value = np.random.randn(1))
        self.w_xh_hy = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w_b_hy = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.h__w_hh_hy, self.w_xh_hy, self.w_b_hy, TRAIN_SLOPE)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.h__w_hh_hy, self.w_xh_hy, self.w_b_hy, self.f_der],
                                variables=[self.x])
        
        
    def optimize_X(self, X, H, w_hy, w_hh, w_xh, b, F_DER):
        h__w_hh_hy = H@(w_hy@w_hh).T
        h__w_hh_hy = h__w_hh_hy.reshape(h__w_hh_hy.size()[0], 1)
        w_xh_hy = w_hy@w_xh
        w_b_hy = b@w_hy.T
        w_b_hy = w_b_hy.reshape(1)
        return self.layer(X, h__w_hh_hy, w_xh_hy, w_b_hy, F_DER)[0]

# # Gain & Cost functions

def score(x, h, w_hy, w_hh, w_xh, b):
    return (h@w_hh.T + x@w_xh.T + b)@w_hy.T

def score_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy):
    return h__w_hh_hy + x@w_xh_hy.T + w_b_hy

def f(x, h, w_hy, w_hh, w_xh, b, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score(x, h, w_hy, w_hh, w_xh, b) + 1)]), 2)

def g(x, h, w_hy, w_hh, w_xh, b, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score(x, h, w_hy, w_hh, w_xh, b) - 1)]), 2)

def g_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy) - 1)]), 2)

def c(x, r):
    return cp.sum_squares(x-r)

def f_derivative(x, h, w_hy, w_hh, w_xh, b, slope):
    return 0.5*slope*((slope*score(x, h, w_hy, w_hh, w_xh, b) + 1)
                        /cp.sqrt((slope*score(x, h, w_hy, w_hh, w_xh, b) + 1)**2 + 1))*(w_hy@w_xh)

funcs = {"f": f, "g": g, "f_derivative": f_derivative, "c": c, "score": score,
         "score_dpp_form": score_dpp_form, "g_dpp_form": g_dpp_form}

# # Model

class MyRNN(torch.nn.Module):
    def __init__(self, x_dim, h_dim, funcs, train_slope, eval_slope, strategic=False, extra=False):
        torch.manual_seed(0)
        np.random.seed(0)
        
        super(MyRNN, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.W_hh = torch.nn.parameter.Parameter(math.sqrt(1/h_dim)*(1-2*torch.rand((h_dim, h_dim), dtype=torch.float64, requires_grad=True)))
        self.W_xh = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand((h_dim, x_dim), dtype=torch.float64, requires_grad=True)))
        self.W_hy = torch.nn.parameter.Parameter(math.sqrt(1/h_dim)*(1-2*torch.rand(h_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(math.sqrt(1/h_dim)*(1-2*torch.rand(h_dim, dtype=torch.float64, requires_grad=True)))
        self.sigmoid = torch.nn.Sigmoid()
        self.strategic = strategic
        self.extra = extra
        self.ccp = CCP(x_dim, h_dim, funcs)
        self.delta = DELTA(x_dim, h_dim, funcs)

    def forward(self, X, evaluation=False):
        batch_size, seq_len, _ = X.size()  # B, 14, 82
        # assert(seq_len == 14)
        X = X.transpose(1,0)
        
        H = torch.zeros((batch_size, h_dim), dtype=torch.float64, requires_grad=False)
        for x in X[:-1]:
            H = self.sigmoid(H@self.W_hh.T + x@self.W_xh.T + self.b)
        
        x = X[-1]
        if self.strategic:
            if evaluation:
                XT = self.ccp.optimize_X(x, H, self.W_hy, self.W_hh, self.W_xh, self.b, self.eval_slope)
                x_opt = XT
            else:
                XT = self.ccp.optimize_X(x, H, self.W_hy, self.W_hh, self.W_xh, self.b, self.train_slope)
                F_DER = self.get_f_ders(XT, H, self.train_slope)
                x_opt = self.delta.optimize_X(x, H, self.W_hy, self.W_hh, self.W_xh, self.b, F_DER)
            H = (H@self.W_hh.T + x_opt@self.W_xh.T + self.b)
        else:
            if self.extra:
                H = self.sigmoid(H@self.W_hh.T + x@self.W_xh.T + self.b)
            else:
                H = (H@self.W_hh.T + x@self.W_xh.T + self.b)
        
        output = H@self.W_hy.T    
        return output
    
    def optimize_X(self, X):
        batch_size, seq_len, _ = X.size()
        X = X.transpose(1,0)
        
        H = torch.zeros((batch_size, h_dim), dtype=torch.float64, requires_grad=False)
        for x in X[:-1]:
            H = self.sigmoid(H@self.W_hh.T + x@self.W_xh.T + self.b)
        
        x = X[-1]
        x = self.ccp.optimize_X(x, H, self.W_hy, self.W_hh, self.W_xh, self.b, self.eval_slope).reshape(1, x.size()[0], x.size()[1])
        X = torch.cat((X[:-1], x), 0)
        return X.transpose(1,0)
    
    def score(self, x, h):
        return (h@self.W_hh.T + x@self.W_xh.T + self.b)@self.W_hy.T
    
    def get_f_ders(self, XT, H, slope):
        W_xhhy = self.W_hy@self.W_xh
        return torch.stack([0.5*slope*((slope*self.score(xt, h) + 1)/torch.sqrt((slope*self.score(xt, h) + 1)**2 + 1))*W_xhhy for xt, h in zip(XT, H)])

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
#                     print("failed")
                
            with torch.no_grad():
#                 try:
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
                    if consecutive_no_improvement >= 10:
                            break
#                 except:
#                     print("failed")
                    
            t2 = time.time()
            if verbose:
                print("------------- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))
        print("training time: {} seconds".format(time.time()-total_time)) 
        return train_errors, val_errors, train_losses, val_losses

# # Train

PATH = "./models/rnn"

EPOCHS = 5
BATCH_SIZE = 16

x_dim, h_dim = XDIM, 10

for seq_len in range(1, 15):
    path = PATH + "/" + str(seq_len)

    X, Y = load_financial_distress_data(seq_len)
    X /= math.sqrt(XDIM)
    X, Xval, Y, Yval = train_test_split(X, Y, test_size=0.4, random_state=10)
    Xval, Xtest, Yval, Ytest = train_test_split(Xval, Yval, test_size=0.5, random_state=10)
    
    if not os.path.exists(path):
        os.makedirs(path)
    pd.DataFrame(torch.flatten(X).numpy()).to_csv(path + '/X.csv')
    pd.DataFrame(torch.flatten(Y).numpy()).to_csv(path + '/Y.csv')
    pd.DataFrame(torch.flatten(Xval).numpy()).to_csv(path + '/Xval.csv')
    pd.DataFrame(torch.flatten(Yval).numpy()).to_csv(path + '/Yval.csv')
    pd.DataFrame(torch.flatten(Xtest).numpy()).to_csv(path + '/Xtest.csv')
    pd.DataFrame(torch.flatten(Ytest).numpy()).to_csv(path + '/Ytest.csv')

    # non-strategic classification
    print("---------- training non-strategically----------")
    non_strategic_model = MyRNN(x_dim, h_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, strategic=False, extra=False)

    non_strategic_model.fit(path, X, Y, Xval, Yval,
                                   opt=torch.optim.Adam, opt_kwargs={"lr": (5e-3)},
                                   batch_size=BATCH_SIZE, epochs=EPOCHS+10, verbose=False,
                                   comment="non_strategic")
    
    # strategic classification
    print("---------- training strategically----------")
    strategic_model = MyRNN(x_dim, h_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, extra=False)

    strategic_model.fit(path, X, Y, Xval, Yval,
                                   opt=torch.optim.Adam, opt_kwargs={"lr": (5e-2)},
                                   batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=False,
                                   comment="strategic")

non_strategic_model = MyRNN(x_dim, h_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, strategic=False, extra=False)
non_strategic_model.load_model(path + "/non_strategic/model.pt")

strategic_model = MyRNN(x_dim, h_dim, funcs, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, extra=False)
strategic_model.load_model(path + "/strategic/model.pt")

accuracies = np.zeros(3)
accuracies[0] = non_strategic_model.evaluate(Xtest, Ytest)

accuracies[1] = strategic_model.evaluate(Xtest, Ytest)

Xtest_opt = non_strategic_model.optimize_X(Xtest)
accuracies[2] = non_strategic_model.evaluate(Xtest_opt, Ytest)

pd.DataFrame(accuracies).to_csv(path + '/results.csv')
