import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as pylt
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from auton_survival.models.cph.dcph_utilities import train_dcph,predict_survival
from auton_survival.metrics import survival_regression_metric
# from estimators_demo_utils import plot_performance_metrics
import pandas as pd
# from .unet_parts import *
import torchvision.models as models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Args = {'load_history': True,
        'epoch':0,
        'learning rate':1e-3,
        'batch_size':64,
        'return_loss':True,
        'save_path':'experiment/CNN/'}


def plot_performance_metrics(results, times):
    """Plot Brier Score, ROC-AUC, and time-dependent concordance index
    for survival model evaluation.

    Parameters
    -----------
    results : dict
      Python dict with key as the evaulation metric
    times : float or list
      A float or list of the times at which to compute
      the survival probability.

    Returns
    -----------
    matplotlib subplots

    """

    colors = ['blue', 'purple', 'orange', 'green']
    gs = gridspec.GridSpec(1, len(results), wspace=0.3)

    for fi, result in enumerate(results.keys()):
        val = results[result]
        x = [str(round(t, 1)) for t in times]
        ax = plt.subplot(gs[0, fi]) # row 0, col 0
        ax.set_xlabel('Time')
        ax.set_ylabel(result)
        ax.set_ylim(0, 1)
        ax.bar(x, val, color=colors[fi])
        plt.xticks(rotation=30)

    plt.savefig(Args['save_path'] + "eval_metric_CNN_model.png")
def increase_censoring(e, t, p, random_seed=0):

    np.random.seed(random_seed)

    uncens = np.where(e == 1)[0]
    mask = np.random.choice([False, True], len(uncens), p=[1-p, p])
    toswitch = uncens[mask]

    e[toswitch] = 0
    t_ = t[toswitch]

    newt = []
    for t__ in t_:
        newt.append(np.random.uniform(1, t__))
    t[toswitch] = newt

    return e, t

def _load_mnist():
    """Helper function to load and preprocess the MNIST dataset.
    The MNIST database of handwritten digits, available from this page, has a
    training set of 60,000 examples, and a test set of 10,000 examples.
    It is a good database for people who want to try learning techniques and
    pattern recognition methods on real-world data while spending minimal
    efforts on preprocessing and formatting [1].
    Please refer to http://yann.lecun.com/exdb/mnist/.
    for the original datasource.
    References
    ----------
    [1]: LeCun, Y. (1998). The MNIST database of handwritten digits.
    http://yann.lecun.com/exdb/mnist/.
    """

    train = torchvision.datasets.MNIST(root='datasets/',
                                       train=True, download=True)
    # print("tar", train.targets[5])
    x = train.data.numpy()
    x = np.expand_dims(x, 1).astype(float)
    t = train.targets.numpy().astype(float) + 1

    e, t = increase_censoring(np.ones(t.shape), t, p=.5)

    return x, t, e

def preprocess_training_data(x, t, e, vsize, val_data, random_seed):

    idx = list(range(x.shape[0]))

    np.random.seed(random_seed)
    np.random.shuffle(idx)

    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).float()
    t_train = torch.from_numpy(t_train).float()
    e_train = torch.from_numpy(e_train).float()

    if val_data is None:

        vsize = int(vsize*x_train.shape[0])
        x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

        x_train = x_train[:-vsize]
        t_train = t_train[:-vsize]
        e_train = e_train[:-vsize]

    else:

        x_val, t_val, e_val = val_data

        x_val = torch.from_numpy(x_val).float()
        t_val = torch.from_numpy(t_val).float()
        e_val = torch.from_numpy(e_val).float()

    return (x_train, t_train, e_train, x_val, t_val, e_val)



class DeepCoxPHTorch(nn.Module):

    def _init_coxph_layers(self, lastdim):
        # have to put it to device since this layer is not init when the model is set
        self.expert = nn.Linear(lastdim, 1, bias=False).to(device)

    def __init__(self, inputdim = None, layers=None, optimizer='Adam'):
        super(DeepCoxPHTorch, self).__init__()

        self.optimizer = optimizer

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0: lastdim = inputdim
        else: lastdim = layers[-1]

        # self.expert = None
        self._init_coxph_layers(1000)
        # CNN insert here
        self.embedding = models.shufflenet_v2_x1_0(pretrained=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # print("check input",x.shape)
        # print("in model",x.device)
        x = self.embedding(x)
        x = self.flatten(x)
        # print(x.shape)
        # print("in model2", x.device)
        # if self.expert is not None:
        #     return self.expert(x)
        # else:
        #     self._init_coxph_layers(x.shape[-1])
        # t = self.expert(x)
        # self.expert = nn.Linear(lastdim, 1, bias=False)
        # print("in model3", t.device)
        return self.expert(x)
        # return x

def to_dframe(t,e):
    d = {'event': e, 'time': t}
    df = pd.DataFrame(data=d)
    return df

def average_re(results):
    return np.array(results['Brier Score']).mean(),np.array(results['Concordance Index']).mean()
if __name__ == '__main__':
    # dowmloading dataset
    x = np.load('data/x.npy',allow_pickle=True)
    t = np.load('data/t.npy', allow_pickle=True)
    e = np.load('data/e.npy', allow_pickle=True)
    x = np.repeat(x,3,axis=1)
    # preprocessing data
    x_tr, x_te, t_tr, t_te, e_tr, e_te = train_test_split(x, t, e, test_size=0.2, random_state=1)
    x_train, t_train, e_train, x_val, t_val, e_val = preprocess_training_data(x, t, e, vsize=0.15, val_data=None,
                                                                              random_seed=0)
    print("Check data type",
          type(x_train),
          type(t_train),
          type(e_train)
          )
    x_train = x_train.to(device)
    t_train = t_train
    e_train = e_train
    x_val = x_val.to(device)
    t_val = t_val
    e_val = e_val
    x_te = torch.from_numpy(x_te).to(torch.float).to(device)
    e_te = torch.from_numpy(e_te).float()
    t_te = torch.from_numpy(t_te).float()

    # device
    print("working on",device)
    # print("check",x_train.device)
    # define the model
    model = DeepCoxPHTorch().to(device)
    print("model summarize",model)
    # train
    (model, breslow_spline), loss = train_dcph(model,
                                               (x_train, t_train, e_train),
                                               (x_val, t_val, e_val),
                                               epochs=Args['epoch'],
                                               lr=Args['learning rate'],
                                               bs=Args['batch_size'],
                                               return_losses=Args['return_loss'],
                                               args=Args)
    # transform the result to data frame in pandas
    d_tr = to_dframe(t_tr, e_tr)
    d_te = to_dframe(t_te, e_te)

    # Define the times for tuning the model hyperparameters and for evaluating the model
    times = np.linspace(2, 9, 8)

    # Obtain survival probabilities for test set
    predictions_te = predict_survival((model, breslow_spline), x_te, t=times)

    # Compute the Brier Score and time-dependent concordance index for the test set to assess model performance
    results = dict()
    results['Brier Score'] = survival_regression_metric('brs', outcomes_train=d_tr, outcomes_test=d_te,
                                                        predictions=predictions_te, times=times)

    results['Concordance Index'] = survival_regression_metric('ctd', outcomes_train=d_tr, outcomes_test=d_te,
                                                              predictions=predictions_te, times=times)
    plot_performance_metrics(results, times)
    br,con = average_re(results)
    print('brier score',br,'con index',con)