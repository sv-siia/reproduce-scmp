import torch
from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def split_data(X, Y, percentage):
    num_val = int(len(X)*percentage)
    return X[num_val:], Y[num_val:], X[:num_val], Y[:num_val]

def shuffle(X, Y):
    torch.manual_seed(0)
    np.random.seed(0)
    data = torch.cat((X, Y), 1)
    data = data[torch.randperm(data.size()[0])]
    X = data[:, :2]
    Y = data[:, 2]
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

def load_credit_default_data():
    torch.manual_seed(0)
    np.random.seed(0)
    url = 'https://raw.githubusercontent.com/ustunb/actionable-recourse/master/examples/paper/data/credit_processed.csv'
    df = pd.read_csv(url)
    df["NoDefaultNextMonth"].replace({0: -1}, inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.drop(['Married', 'Single', 'Age_lt_25', 'Age_in_25_to_40', 'Age_in_40_to_59', 'Age_geq_60'], axis=1)
    scaler = StandardScaler()
    df.loc[:, df.columns != "NoDefaultNextMonth"] = scaler.fit_transform(df.drop("NoDefaultNextMonth", axis=1))
    fraud_df = df.loc[df["NoDefaultNextMonth"] == -1]
    non_fraud_df = df.loc[df["NoDefaultNextMonth"] == 1][:6636]
    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
    df = normal_distributed_df.sample(frac=1).reset_index(drop=True)
    Y, X = df.iloc[:, 0].values, df.iloc[:, 1:].values
    return torch.from_numpy(X), torch.from_numpy(Y)


def load_financial_distress_data(seq_len=14):
    assert(1 <= seq_len <= 14)
    torch.manual_seed(0)
    np.random.seed(0)

    data = pd.read_csv("./dataset/Financial_Distress.csv")

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


def load_spam_dataset():
    torch.manual_seed(0)
    np.random.seed(0)
    path = r"./dataset/IS_journal_tip_spam.arff"
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    most_disc = ['qTips_plc', 'rating_plc', 'qEmail_tip', 'qContacts_tip', 'qURL_tip', 'qPhone_tip', 'qNumeriChar_tip', 'sentistrength_tip', 'combined_tip', 'qWords_tip', 'followers_followees_gph', 'qunigram_avg_tip', 'qTips_usr', 'indeg_gph', 'qCapitalChar_tip', 'class1']
    df = df[most_disc]
    df["class1"].replace({b'spam': -1, b'notspam': 1}, inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)

    Y = df['class1'].values
    X = df.drop('class1', axis = 1).values
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return torch.from_numpy(X), torch.from_numpy(Y)