import os
import math
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.io import arff
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from strategic_classification.models.vanila.vanilaccp import VanilaCCP
from strategic_classification.models.vanila.vaniladelta import VanilaDelta

TRAIN_SLOPE = 1
EVAL_SLOPE = 5
SEED = 0

class MyStrategicModel(torch.nn.Module):
    def __init__(self, x_dim, train_slope, eval_slope, scale, strategic=False):
        torch.manual_seed(0)
        np.random.seed(0)

        super(MyStrategicModel, self).__init__()
        self.x_dim = x_dim
        self.train_slope, self.eval_slope = train_slope, eval_slope
        self.w = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(x_dim, dtype=torch.float64, requires_grad=True)))
        self.b = torch.nn.parameter.Parameter(math.sqrt(1/x_dim)*(1-2*torch.rand(1, dtype=torch.float64, requires_grad=True)))
        self.strategic = strategic
        self.ccp = VanilaCCP(self.x_dim, scale)
        self.delta = VanilaDelta(self.x_dim, scale)

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


def load_spam_data():
    torch.manual_seed(0)
    np.random.seed(0)
    path = "dataset/IS_journal_tip_spam.arff"
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    most_disc = ['qTips_plc', 'rating_plc', 'qEmail_tip', 'qContacts_tip', 'qURL_tip', 'qPhone_tip', 'qNumeriChar_tip', 'sentistrength_tip', 'combined_tip', 'qWords_tip', 'followers_followees_gph', 'qTips_usr', 'indeg_gph', 'qCapitalChar_tip', 'class1']
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


def load_credit_default_data():
    torch.manual_seed(0)
    np.random.seed(0)
    url = 'https://raw.githubusercontent.com/ustunb/actionable-recourse/master/examples/paper/data/credit_processed.csv'
    df = pd.read_csv(url)
    df["NoDefaultNextMonth"].replace({0: -1}, inplace=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    df = df.drop(['Married', 'Single', 'Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60'], axis = 1)

    fraud_df = df.loc[df["NoDefaultNextMonth"] == -1]
    non_fraud_df = df.loc[df["NoDefaultNextMonth"] == 1][:6636]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    df = normal_distributed_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    scaler = StandardScaler()
    df.loc[:, df.columns != "NoDefaultNextMonth"] = scaler.fit_transform(df.drop("NoDefaultNextMonth", axis=1)) 
    Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_financial_distress_data():
    torch.manual_seed(0)
    np.random.seed(0)
    data = pd.read_csv("dataset/financial_distress.csv")

    data = data[data.columns.drop(list(data.filter(regex='x80')))] # Since it is a categorical feature with 37 features.
    x_dim = len(data.columns) - 3
    data.drop(['Time'], axis=1, inplace=True)

    data_grouped = data.groupby(['Company']).last()

    scaler = StandardScaler()
    data_grouped.loc[:, data_grouped.columns != "Financial Distress"] = scaler.fit_transform(data_grouped.drop("Financial Distress", axis=1))

    # Shuffle dataframe rows
    data_grouped = data_grouped.sample(frac=1, random_state=SEED).reset_index(drop=True)

    Y, X = data_grouped.iloc[:, 0].values, data_grouped.iloc[:, 1:].values
    for y in range(0,len(Y)): # Coverting target variable from continuous to binary form
        if Y[y] < -0.5:
              Y[y] = -1
        else:
              Y[y] = 1
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    return torch.from_numpy(X), torch.from_numpy(Y)


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


def datasets():
    training_datas = []
    # distress 
    X, Y = load_financial_distress_data()
    X, Y, Xval, Yval = split_data(X, Y, 0.4)
    Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)
    training_datas.append({"X": X,
                            "Y": Y,
                            "Xval": Xval,
                            "Yval": Yval,
                            "Xtest": Xtest,
                            "Ytest": Ytest,
                            "epochs": 7,
                            "batch_size": 24,
                            "name": "distress"})

    # credit data
    X, Y = load_credit_default_data()
    X, Y = X[:3000], Y[:3000]
    X, Y, Xval, Yval = split_data(X, Y, 0.4)
    Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)
    training_datas.append({"X": X,
                            "Y": Y,
                            "Xval": Xval,
                            "Yval": Yval,
                            "Xtest": Xtest,
                            "Ytest": Ytest,
                            "epochs": 7,
                            "batch_size": 64, 
                            "name": "credit"})

    # spam dataset
    X, Y = load_spam_data()
    X, Y, Xval, Yval = split_data(X, Y, 0.4)
    Xval, Yval, Xtest, Ytest = split_data(Xval, Yval, 0.5)
    training_datas.append({"X": X,
                            "Y": Y,
                            "Xval": Xval,
                            "Yval": Yval,
                            "Xtest": Xtest,
                            "Ytest": Ytest,
                            "epochs": 7,
                            "batch_size": 128, 
                            "name": "spam"})
    return training_datas


def train_vanila(epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/vanila"):
    scales = [1/2, 1, 2]    

    for training_data in datasets():
        path = model_checkpoint_path + "/" + training_data["name"]
        
        # load dataset
        X = training_data["X"]
        Y = training_data["Y"]
        Xval = training_data["Xval"]
        Yval = training_data["Yval"]
        Xtest = training_data["Xtest"]
        Ytest = training_data["Ytest"]
        
        # save dataset splits
        if not os.path.exists(path):
            os.makedirs(path)
        pd.DataFrame(X.numpy()).to_csv(path + '/X.csv')
        pd.DataFrame(Y.numpy()).to_csv(path + '/Y.csv')
        pd.DataFrame(Xval.numpy()).to_csv(path + '/Xval.csv')
        pd.DataFrame(Yval.numpy()).to_csv(path + '/Yval.csv')
        pd.DataFrame(Xtest.numpy()).to_csv(path + '/Xtest.csv')
        pd.DataFrame(Ytest.numpy()).to_csv(path + '/Ytest.csv')
        
        # training parameters
        x_dim = len(X[0])
        epochs = training_data["epochs"]
        batch_size = training_data["batch_size"]
        
        for scale in scales:
            path = path + "/" + str(scale)
            print("------------------------- {}, {} -------------------------".format(training_data["name"], scale))
            
            # non-strategic classification
            print("---------- training non-strategically----------")
            non_strategic_model = MyStrategicModel(x_dim, TRAIN_SLOPE, EVAL_SLOPE, scale=scale, strategic=False)
            non_strategic_model.fit(path, X, Y, Xval, Yval,
                                    opt=torch.optim.Adam, opt_kwargs={"lr": (1e-1)},
                                    batch_size=batch_size, epochs=epochs, verbose=False,
                                comment="non_strategic")
            
            non_strategic_model = MyStrategicModel(x_dim, TRAIN_SLOPE, EVAL_SLOPE, scale=scale, strategic=False)
            non_strategic_model.load_model(path + "/non_strategic/model.pt")
            non_strategic_model.normalize_weights()
            
            
            # strategic classification
            print("---------- training strategically----------")
            strategic_model = MyStrategicModel(x_dim, TRAIN_SLOPE, EVAL_SLOPE, scale=scale, strategic=True)
            strategic_model.fit(path, X, Y, Xval, Yval,
                                opt=torch.optim.Adam, opt_kwargs={"lr": 5*(1e-1)},
                                batch_size=batch_size, epochs=epochs, verbose=False, 
                            comment="strategic") 
            
            strategic_model = MyStrategicModel(x_dim, TRAIN_SLOPE, EVAL_SLOPE, scale=scale, strategic=True)
            strategic_model.load_model(path + "/strategic/model.pt")
            
            # calculate results
            accuracies = np.zeros(3)
            # non strategic model & non strategic data
            accuracies[0] = (non_strategic_model.evaluate(Xtest, Ytest))
            # strategic model & strategic data
            accuracies[1] = (strategic_model.evaluate(Xtest, Ytest))
            # non strategic model & strategic data
            Xtest_opt = non_strategic_model.optimize_X(Xtest, evaluation=True)
            accuracies[2] = (non_strategic_model.evaluate(Xtest_opt, Ytest))
            
            pd.DataFrame(accuracies).to_csv(path + '/results.csv')
    