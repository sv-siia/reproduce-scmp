import cvxpy as cp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
import math
import os
from datetime import datetime
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from strategic_classification.utils.data_utils import split_data, shuffle
from strategic_classification.models.manifold.manifoldcpp import CCP, CCP_MANIFOLD
from strategic_classification.models.manifold.manifolddelta import DELTA, DELTA_MANIFOLD
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

XDIM=2

TRAIN_SLOPE = 1
EVAL_SLOPE = 1
X_LOWER_BOUND = -30
X_UPPER_BOUND = 30


# Dataset
def gen_custom_normal_data(N, x_dim, pos_mean, pos_std, neg_mean, neg_std):
    torch.manual_seed(0)
    np.random.seed(0)
    pos_samples_num = N//2
    neg_samples_num = N - pos_samples_num
    posX = torch.randn((pos_samples_num, x_dim))*pos_std + pos_mean
    negX = torch.randn((neg_samples_num, x_dim))*neg_std + neg_mean

    X = torch.cat((posX, negX), 0)
    Y = torch.unsqueeze(torch.cat((torch.ones(len(posX)), -torch.ones(len(negX))), 0), 1)

    X, Y = shuffle(X, Y)
    return X, Y

def gen_custom_sin_data(N, shuff=True):

    def func(x):
        return -(x**2)

    torch.manual_seed(0)
    np.random.seed(0)
    pos_samples_num = N//2
    neg_samples_num = N - pos_samples_num

    posX = torch.linspace(-5, 0, pos_samples_num)
    posX = torch.stack([posX, func(posX)])
    posX = torch.transpose(posX, 1, 0)

    negX = torch.linspace(0, 5, neg_samples_num)
    negX = torch.stack([negX, func(negX)])
    negX = torch.transpose(negX, 1, 0)

    X = torch.cat((posX, negX), 0)
    Y = torch.unsqueeze(torch.cat((torch.ones(len(posX)), -torch.ones(len(negX))), 0), 1)
    if shuff:
        X, Y = shuffle(X, Y)
    else:
        Y = Y[:, 0]
    return X, Y

def generate_manifold_data(N=2000, x_dim=2, h_dim=1, h_dim2=20):
    X, Y = gen_custom_sin_data(N)
    X, Y, Xval, Yval = split_data(X, Y, 0.4)
    Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)
    path = "./models/manifold/CAE"
    cae = CAE(x_dim, h_dim, h_dim2, 100)
    cae.fit(path, X, Xval, opt=torch.optim.Adam, opt_kwargs={"lr": (5e-2)}, batch_size=64, epochs=3000, verbose=True, comment='manifold')
    B_SPANS = cae.get_spans(X)
    B_SPANSval = cae.get_spans(Xval)
    B_SPANStest = cae.get_spans(Xtest)
    return X, Y, Xval, Yval, Xtest, Ytest, B_SPANS, B_SPANSval, B_SPANStest

# CAE
class CAE(nn.Module):
    def __init__(self, x_dim, h_dim, h2_dim, lamb=0):
        torch.manual_seed(0)
        np.random.seed(0)
        super(CAE, self).__init__()

        self.lamb = lamb
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.h2_dim = h2_dim
        self.fc1 = nn.Linear(x_dim, h2_dim, bias = True) # Encoder
        self.fc3 = nn.Linear(h2_dim, h_dim, bias = True)
        self.fc4 = nn.Linear(h_dim, h2_dim, bias = True)
        self.fc5 = nn.Linear(h2_dim, h2_dim, bias = True)
        self.fc7 = nn.Linear(h2_dim, x_dim, bias = True)

        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        o1 = self.sigmoid(self.fc1(x))
        o2 = self.sigmoid(self.fc3(o1))
        return o2

    def decoder(self, z):
        o1 = self.sigmoid(self.fc4(z))
        o2 = self.sigmoid(self.fc5(o1))
        return self.fc7(o2)

    def forward(self, x):
        h1 = self.encoder(x)
        h2 = self.decoder(h1)
        return h1, h2

    def save_model(self, path, comment=None):
        if comment is not None:
            path += "/" + comment

        filename = path + "/model.pt"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

    def get_spans(self, X):
        def func(x):
            return self.forward(x)[0]

        B_SPANS = []
        for x in X:
            J = jacobian(func, x)
            U, S, _ = torch.svd(torch.transpose(J, 0, 1))
            B_span = U
            B_SPANS.append(B_span)
        return torch.stack(B_SPANS)

    def contractive_loss(self, x):
        def func(x):
            return self.encoder(x)
        J = jacobian(func, x, create_graph=True)
        c_loss = torch.norm(J, 2)**2
        return c_loss

    def reconstruction_loss(self, x, x_recons):
        mse_loss = nn.MSELoss(size_average = True)
        r_loss = mse_loss(x_recons, x)
        return r_loss

    def loss(self, x, x_recons, h):
        r_loss = self.reconstruction_loss(x, x_recons)
        c_loss = self.contractive_loss(x)
        return r_loss, c_loss

    def fit(self, path, X, Xval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, comment=None):
        train_dset = TensorDataset(X, torch.ones(len(X)))
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
        opt = opt(self.parameters(), **opt_kwargs)

        best_val_loss = 100

        for epoch in range(epochs):
            total_r_loss = 0
            total_c_loss = 0
            train_loss = 0
            self.train()

            for idx, (Xbatch, _) in enumerate(train_loader):
                Xbatch = Variable(Xbatch)
                opt.zero_grad()

                hidden_representation, recons_x = self.forward(Xbatch)
                r_loss, c_loss = self.loss(Xbatch, recons_x, hidden_representation)
                l = r_loss + self.lamb*c_loss
                l.backward()
                train_loss += l.item()
                total_r_loss += r_loss.item()
                total_c_loss += c_loss.item()
                opt.step()

            hidden_representation, recons_x = self.forward(Xval)
            r_loss, c_loss = self.loss(Xval, recons_x, hidden_representation)
            l = r_loss + self.lamb*c_loss
            if l.item() < best_val_loss:
                self.save_model(path, comment)
                best_val_loss = l.item()
                print("model saved!")

            if verbose:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                     epoch, l.item()), " reconstruction loss: ", r_loss.item(), "contractive_loss: ", c_loss.item())

# Model
class MyManifold(torch.nn.Module):
    def __init__(self, x_dim, h_dim, train_slope, eval_slope, strategic=False, manifold=False):
        torch.manual_seed(0)
        np.random.seed(0)
        super(MyManifold, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(1, dtype=torch.float64, requires_grad=True)))
        self.strategic = strategic
        self.manifold = manifold
        if self.manifold:
            self.ccp_train = CCP_MANIFOLD(self.x_dim, self.h_dim, )
            self.delta = DELTA_MANIFOLD(self.x_dim, self.h_dim, )
        else:
            self.ccp_train = CCP(self.x_dim, self.h_dim, )
            self.delta = DELTA(self.x_dim, self.h_dim, )

        self.ccp_test = CCP_MANIFOLD(self.x_dim, self.h_dim, )

    def forward(self, X, B_SPANS, evaluation=False):
        if self.strategic:            
            if evaluation:
                XT = self.ccp_train.optimize_X(X, self.w, self.b, B_SPANS, self.eval_slope)
                X_opt = XT
            else:
                XT = self.ccp_train.optimize_X(X, self.w, self.b, B_SPANS, self.train_slope)
                F_DER = self.get_f_ders(XT, self.train_slope)
                X_opt = self.delta.optimize_X(X, self.w, self.b, F_DER, B_SPANS) # Xopt should be equal to XT but we do it again for the gradients

            output = self.score(X_opt)
        else:
            output = self.score(X)        
        return output

    def optimize_X(self, X, B_SPANS):
        return self.ccp_test.optimize_X(X, self.w, self.b, B_SPANS, self.eval_slope)

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

    def evaluate(self, X, B_SPANS, Y):      
        return self.calc_accuracy(Y, self.forward(X, B_SPANS, evaluation=True))

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

    def fit(self, path, X, B_SPANS, Y, Xval, B_SPANSval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None, comment=None):
        train_dset = TensorDataset(X, B_SPANS, Y)
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
            for Xbatch, B_SPANSbatch, Ybatch in train_loader:
                opt.zero_grad()
                Ybatch_pred = self.forward(Xbatch, B_SPANSbatch)
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

            with torch.no_grad():
                Yval_pred = self.forward(Xval, B_SPANSval, evaluation=True)
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

            t2 = time.time()
            if verbose:
                print("----- epoch %03d / %03d | time: %03d sec | loss: %3.5f | err: %3.5f" % (epoch + 1, epochs, t2-t1, val_losses[-1], val_errors[-1]))
        print("training time: {} seconds".format(time.time()-total_time)) 
        return train_errors, val_errors, train_losses, val_losses


def train_manifold(dataset = None, epochs: int = 6, batch_size: int = 64, model_checkpoint_path: str = "./models/manifold"):
    """
    Train manifold models with the given parameters.
    """
    # Data generation
    x_dim = 2
    h_dim = 1
    h_dim2 = 20
    X, Y, Xval, Yval, Xtest, Ytest, B_SPANS, B_SPANSval, B_SPANStest = generate_manifold_data(N=2000, x_dim=x_dim, h_dim=h_dim, h_dim2=h_dim2)

    print(X.size(), Y.size())
    print(X.size(), Xval.size())
    print("percent of positive samples: {}%".format(100 * len(Y[Y == 1]) / len(Y)))

    now = datetime.now()
    PATH = model_checkpoint_path + "/" + now.strftime("%d-%m-%Y_%H-%M-%S")

    # non-strategic classification
    print("---------- training non-strategically----------")
    non_strategic_model = MyManifold(x_dim, h_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=False)
    non_strategic_model.fit(PATH, X, B_SPANS, Y, Xval, B_SPANSval, Yval,
                            opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-1)},
                            batch_size=batch_size, epochs=epochs, verbose=True,
                            comment="non_strategic")

    # strategic classification (naive)
    print("---------- training strategically----------")
    strategic_model_naive = MyManifold(x_dim, h_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, manifold=False)
    strategic_model_naive.fit(PATH, X, B_SPANS, Y, Xval, B_SPANSval, Yval,
                             opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-1)},
                             batch_size=batch_size, epochs=epochs, verbose=True,
                             comment="strategic_naive")

    # strategic classification (manifold)
    print("---------- training strategically----------")
    strategic_model_man = MyManifold(x_dim, h_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, manifold=True)
    strategic_model_man.fit(PATH, X, B_SPANS, Y, Xval, B_SPANSval, Yval,
                            opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-1)},
                            batch_size=batch_size, epochs=epochs, verbose=True,
                            comment="strategic_man")

    # Load models for evaluation
    non_strategic_model = MyManifold(x_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=False)
    non_strategic_model.load_model(PATH + "/non_strategic/model.pt")

    strategic_model_naive = MyManifold(x_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, manifold=False)
    strategic_model_naive.load_model(PATH + "/strategic_naive/model.pt")

    strategic_model_man = MyManifold(x_dim, TRAIN_SLOPE, EVAL_SLOPE, strategic=True, manifold=True)
    strategic_model_man.load_model(PATH + "/strategic_man/model.pt")

    # calculate results
    accuracies = np.zeros(4)
    accuracies[0] = (non_strategic_model.evaluate(Xtest, B_SPANStest, Ytest))
    Xtest_opt = strategic_model_naive.optimize_X(Xtest, B_SPANStest)
    test_scores = strategic_model_naive.score(Xtest_opt)
    accuracies[1] = (strategic_model_naive.calc_accuracy(Ytest, test_scores))
    Xtest_opt = strategic_model_man.optimize_X(Xtest, B_SPANStest)
    test_scores = strategic_model_man.score(Xtest_opt)
    accuracies[2] = (strategic_model_man.calc_accuracy(Ytest, test_scores))
    Xtest_opt = non_strategic_model.optimize_X(Xtest, B_SPANStest)
    accuracies[3] = (non_strategic_model.evaluate(Xtest_opt, B_SPANStest, Ytest))
    pd.DataFrame(accuracies).to_csv(PATH + '/results.csv')