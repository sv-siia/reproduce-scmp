import os
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.strategic_classification.models.ccp import CCP
from src.strategic_classification.models.delta import DELTA
torch.set_default_dtype(torch.float64)


class MyRNN(torch.nn.Module):
    """A simple RNN model with strategic classification capabilities."""
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
                        if consecutive_no_improvement >= 10:
                                break
                except:
                    print("failed")
                    
            t2 = time.time()
            if verbose:
                print("------------- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))

        print("training time: {} seconds".format(time.time()-total_time)) 
        return train_errors, val_errors, train_losses, val_losses
