import os
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from strategic_classification.models.recourse.recoursecpp import CCPRecourse as CCP
from strategic_classification.models.recourse.recoursedelta import DeltaRecourse as DELTA
from strategic_classification.utils.gain_and_cost_func import score, f, g, f_derivative
from strategic_classification.utils.data_utils import load_credit_default_data, split_data
from strategic_classification.config.constants import *
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

XDIM = 11
COST = 1/XDIM
TRAIN_SLOPE = 1



class MyRecourse(torch.nn.Module):
    def __init__(self, x_dim, train_slope, eval_slope, strategic=False, lamb=0):
        torch.manual_seed(0)
        np.random.seed(0)
        super(MyRecourse, self).__init__()
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))
        self.sigmoid = torch.nn.Sigmoid()
        self.strategic = strategic
        self.lamb = lamb
        self.ccp = CCP(x_dim)
        self.delta = DELTA(x_dim)

    def forward(self, X, evaluation=False):
        if evaluation:
            XT = self.ccp.optimize_X(X, self.w, self.b, self.eval_slope)
            X_opt = XT
        else:
            XT = self.ccp.optimize_X(X, self.w, self.b, self.train_slope)
            F_DER = self.get_f_ders(XT, self.train_slope)
            X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER)
        recourse = self.calc_recourse(X, X_opt)
        if self.strategic:
            output = self.score(X_opt)
        else:
            output = self.score(X)
        return output, recourse

    def optimize_X(self, X, evaluation=False):
        slope = self.eval_slope if evaluation else self.train_slope
        return self.ccp.optimize_X(X, self.w, self.b, slope)

    def score(self, x):
        return x @ self.w + self.b

    def get_f_ders(self, XT, slope):
        return torch.stack([0.5 * slope * ((slope * self.score(xt) + 1) / torch.sqrt((slope * self.score(xt) + 1) ** 2 + 1)) * self.w for xt in XT])

    def evaluate(self, X, Y):
        scores, _ = self.forward(X, evaluation=True)
        Y_pred = torch.sign(scores)
        num = len(Y)
        temp = Y - Y_pred
        acc = len(temp[temp == 0]) * 1. / num
        return acc

    def calc_recourse(self, X, X_opt):
        S = self.score(X)
        is_neg = self.sigmoid(-S)
        S = self.score(X_opt)
        is_not_able_to_be_pos = self.sigmoid(-S)
        recourse_penalty = is_neg * is_not_able_to_be_pos
        return 1 - torch.mean(recourse_penalty)

    def loss(self, Y, Y_pred, recourse):
        if self.lamb > 0:
            return torch.mean(torch.clamp(1 - Y_pred * Y, min=0)) + self.lamb * (1 - recourse)
        else:
            return torch.mean(torch.clamp(1 - Y_pred * Y, min=0))

    def save_model(self, X, Y, Xval, Yval, Xtest, Ytest, train_errors, val_errors, train_losses, val_losses, val_recourses, info, path, comment=None):
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
        pd.DataFrame(Xtest.numpy()).to_csv(path + '/Xtest.csv')
        pd.DataFrame(Ytest.numpy()).to_csv(path + '/Ytest.csv')
        pd.DataFrame(np.array(train_errors)).to_csv(path + '/train_errors.csv')
        pd.DataFrame(np.array(val_errors)).to_csv(path + '/val_errors.csv')
        pd.DataFrame(np.array(train_losses)).to_csv(path + '/train_losses.csv')
        pd.DataFrame(np.array(val_losses)).to_csv(path + '/val_losses.csv')
        pd.DataFrame(np.array(val_recourses)).to_csv(path + '/val_recourses.csv')
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
        val_recourses = []
        best_val_error = 1
        consecutive_no_improvement = 0
        now = datetime.now()
        path = "./models/recourse/" + now.strftime("%d-%m-%Y_%H-%M-%S")
        total_time = time.time()
        for epoch in range(epochs):
            t1 = time.time()
            batch = 1
            train_losses.append([])
            train_errors.append([])
            for Xbatch, Ybatch in train_loader:
                opt.zero_grad()
                Ybatch_pred, recourse = self.forward(Xbatch)
                l = self.loss(Ybatch, Ybatch_pred, recourse)
                l.backward()
                opt.step()
                train_losses[-1].append(l.item())
                if calc_train_errors:
                    with torch.no_grad():
                        e = self.evaluate(Xbatch, Ybatch)
                        train_errors[-1].append(1 - e)
                if verbose:
                    print("batch %03d / %03d | loss: %3.5f | recourse: %3.5f" %
                          (batch, len(train_loader), np.mean(train_losses[-1]), recourse))
                batch += 1
                if callback is not None:
                    callback()
            with torch.no_grad():
                Yval_pred, val_recourse = self.forward(Xval, evaluation=True)
                val_recourse = val_recourse.item()
                val_loss = self.loss(Yval, Yval_pred, val_recourse).item()
                val_losses.append(val_loss)
                val_error = 1 - self.evaluate(Xval, Yval)
                val_errors.append(val_error)
                val_recourses.append(val_recourse)
                if val_error < best_val_error:
                    consecutive_no_improvement = 0
                    best_val_error = val_error
                    if self.strategic:
                        info = "training time in seconds: {}\nepoch: {}\nbatch size: {}\ntrain slope: {}\neval slope: {}\nlearning rate: {}\nvalidation loss: {}\nvalidation error: {}\nrecourse: {}".format(
                            time.time() - total_time, epoch, batch_size, self.train_slope, self.eval_slope, opt_kwargs["lr"], val_loss, val_error, val_recourse)
                        self.save_model(X, Y, Xval, Yval, Xtest, Ytest, train_errors, val_errors, train_losses, val_losses, val_recourses, info, path, comment)
                        print("model saved!")
                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 6:
                        break
            t2 = time.time()
            if verbose:
                print("----- epoch %03d / %03d | time: %03d sec | loss: %3.5f | recourse: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2 - t1, val_losses[-1], val_recourses[-1], val_errors[-1]))
        print("training time: {} seconds".format(time.time() - total_time))
        return train_errors, val_errors, train_losses, val_losses

def train_recourse(dataset_path: str, epochs: int = 10, batch_size: int = 64, model_checkpoint_path: str = "models/recourse"):
    """
    Train a recourse model with the given parameters.
    Args:
        dataset_path (str): Path to the dataset file.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model_checkpoint_path (str): Directory to save the trained models and results.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    # Data generation
    X, Y = load_credit_default_data()
    X, Y = X[:3000], Y[:3000]
    assert(len(X[0]) == XDIM)
    X, Y, Xval, Yval = split_data(X, Y, 0.5)
    Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)
    print("percent of positive samples: {}%".format(100 * len(Y[Y == 1]) / len(Y)))
    x_dim = XDIM
    lambda_range = torch.logspace(start=0, end=0.3, steps=5)
    for lamb in lambda_range:
        print("---------- training strategically----------")
        print("lambda: ", lamb.item())
        strategic_model = MyRecourse(x_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, lamb=lamb)
        fit_res_strategic = strategic_model.fit(
            X, Y, Xval, Yval, Xtest, Ytest,
            opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-2)},
            batch_size=batch_size, epochs=epochs, verbose=True, calc_train_errors=False,
            comment="recourse_" + str(lamb.item())
        )


