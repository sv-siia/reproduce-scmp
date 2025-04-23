import os
import time
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from strategic_classification.models.batched.batcheddelta import BatchedDelta
from strategic_classification.models.batched.batchedccp import BatchedCCP


class Burden(torch.nn.Module):
    def __init__(self, x_dim, batch_size, train_slope, eval_slope, scale, strategic=False):
        torch.manual_seed(0)
        np.random.seed(0)

        super(Burden, self).__init__()
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(1, dtype=torch.float64, requires_grad=True)))
        self.strategic = strategic
        self.ccp = BatchedCCP(self.x_dim, self.batch_size, scale)
        self.delta = BatchedDelta(self.x_dim, scale)
        self.ccp_time = 0
        self.total_time = 0

    def forward(self, X, evaluation=False):
        if self.strategic:
            if evaluation:
                t1 = time.time()
                XT = self.ccp.optimize_X(X, self.w, self.b, self.eval_slope)
                self.ccp_time += time.time()-t1
                X_opt = XT
            else:
                t1 = time.time()
                XT = self.ccp.optimize_X(X, self.w, self.b, self.train_slope)
                self.ccp_time += time.time()-t1
                F_DER = self.get_f_ders(XT, self.train_slope)
                X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER) # Xopt should be equal to XT but we do it again for the gradients
            output = self.score(X_opt)
        else:
            output = self.score(X)        
        return output
    
    def optimize_X(self, X, evaluation=False):
        slope = self.eval_slope if evaluation else self.train_slope
        return self.ccp.optimize_X(X, self.w, self.b, slope)
    
    def normalize_weights(self):
        with torch.no_grad():
            norm = torch.sqrt(torch.sum(self.w**2) + self.b**2)
            self.w /= norm
            self.b /= norm

    def score(self, x):
        return x@self.w + self.b
    
    def get_f_ders(self, XT, slope):
        nablas = 0.5*slope*((slope*self.score(XT) + 1)/torch.sqrt((slope*self.score(XT) + 1)**2 + 1))
        return torch.reshape(nablas, (len(nablas), 1))@torch.reshape(self.w, (1, len(self.w)))

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
        test_dset = TensorDataset(Xval, Yval)
        test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=True)
        
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
                total_loss = 0
                total_error = 0
                batch = 0
                for Xbatch, Ybatch in test_loader:
#                     try:
                    Yval_pred = self.forward(Xbatch, evaluation=True)
                    val_loss = self.loss(Ybatch, Yval_pred).item()
                    total_loss += val_loss
                    val_error = 1-self.calc_accuracy(Ybatch, Yval_pred)
                    total_error += val_error
                    batch += 1
#                     except:
#                         print("failed")
                        
                avg_loss = total_loss/batch
                avg_error = total_error/batch
                val_losses.append(avg_loss)
                val_errors.append(avg_error)
                if avg_error < best_val_error:
                        consecutive_no_improvement = 0
                        best_val_error = avg_error
                        info = "training time in seconds: {}\nepoch: {}\nbatch size: {}\ntrain slope: {}\neval slope: {}\nlearning rate: {}\nvalidation loss: {}\nvalidation error: {}\n".format(
                        time.time()-total_time, epoch, batch_size, self.train_slope, self.eval_slope, opt_kwargs["lr"], avg_loss, avg_error)
                        self.save_model(train_errors, val_errors, train_losses, val_losses, info, path, comment)
                        print("model saved!")

                else:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= 4:
                        break
                    
            t2 = time.time()
            if verbose:
                print("------------- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))
        
        self.total_time = time.time()-total_time
        print("training time: {} seconds".format(self.total_time)) 
        return train_errors, val_errors, train_losses, val_losses


def gen_sklearn_data(x_dim, N, informative_frac=1, shift_range=1, scale_range=1, noise_frac=0.01):
    torch.manual_seed(0)
    np.random.seed(0)
    n_informative = int(informative_frac*x_dim)
    n_redundant = x_dim - n_informative
    shift_arr = shift_range*np.random.randn(x_dim)
    scale_arr = scale_range*np.random.randn(x_dim)
    X, Y = make_classification(n_samples=N, n_features=x_dim, n_informative=n_informative, n_redundant=n_redundant,
                               flip_y=noise_frac, shift=shift_arr, scale=scale_arr, random_state=0)
    Y[Y == 0] = -1
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return torch.from_numpy(X), torch.from_numpy(Y)


def split_data(X, Y, percentage):
    num_val = int(len(X)*percentage)
    return X[num_val:], Y[num_val:], X[:num_val], Y[:num_val]


def train_batched(epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/burden"):
    x_dim = 5
    scale = 1
    TRAIN_SLOPE = 1
    EVAL_SLOPE = 5
    X, Y = gen_sklearn_data(x_dim, 1024)
    X, Y, Xval, Yval = split_data(X, Y, 0.25)
    print(Xval.size())
    print("percent of positive samples: {}%".format(100 * len(Y[Y == 1]) / len(Y)))

    total = []
    ccp = []
    for batch_size in (2**np.arange(9)).tolist():
        path = model_checkpoint_path + "/" + str(batch_size)
        strategic_model = Burden(x_dim, batch_size, TRAIN_SLOPE, EVAL_SLOPE, scale=scale, strategic=True)
        strategic_model.fit(path, X, Y, Xval, Yval,
                            opt=torch.optim.Adam, opt_kwargs={"lr": (1e-1)},
                            batch_size=batch_size, epochs=epochs, verbose=True,
                        comment="batched")
        
        total_time = strategic_model.total_time
        ccp_time = strategic_model.ccp_time
        total.append(total_time)
        ccp.append(ccp_time)
        pd.DataFrame(np.array(total)).to_csv(path + '/total_timing_results.csv')
        pd.DataFrame(np.array(ccp)).to_csv(path + '/ccp_timing_results.csv')
