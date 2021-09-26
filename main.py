import os
import argparse
import math
import time

import torch
import torch.nn as nn
import numpy as np

from tensorboardX import SummaryWriter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models import DSLSTM
from utils import Data_utility
from Optim import Optim

def make_dir(path, dic_name):
    path = os.path.join(path, dic_name)
    is_dir_exist = os.path.exists(path)
    if is_dir_exist:
        print("----Dic existed----")
    else:
        os.mkdir(path)
        print("----Dic created successfully----")
    return path

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, mode):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    correlation = 0
    predict = None
    truth = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = X.cuda()
        Y = Y.cuda()
        output = model(X)

        scale = data.scale.expand(output.size(0), data.m)
        scale = scale.cuda()
        total_loss += evaluateL2(output * scale, Y * scale).cpu().data.numpy()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).cpu().data.numpy()
        n_samples += (output.size(0) * data.m)
        output = output.cpu().detach().numpy()
        Y = Y.cpu().numpy()
        if predict is None:
            predict = output
            truth = Y
        else:
            predict = np.concatenate((predict, output))
            truth = np.concatenate((truth, Y))
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    n = 0
    for i in range(data.m):
        corr, p = pearsonr(predict[:, i].flatten(), truth[:, i].flatten())
        if corr < 1:
            n += 1
            correlation += corr
        else:
            print(i)
    correlation = correlation / n
    if mode == "test":
        np.savetxt(os.path.join(path_result, "predict.csv"), predict)
        np.savetxt(os.path.join(path_result, "truth.csv"), truth)
        with open(os.path.join(path_result, "result.txt"), "w") as result_file:
            result_file.write("RSE:{}\r\n".format(rse))
            result_file.write("RAE:{}\r\n".format(rae))
            result_file.write(("CORR:{}\r\n".format(correlation)))

    return rse, rae, correlation



def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        X = X.cuda()
        Y = Y.cuda()
        model.zero_grad()
        output = model(X)
        output = torch.squeeze(output)
        scale = data.scale.expand(output.size(0), data.m)
        scale = scale.cuda()
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.cpu().data.numpy()
        n_samples += (output.size(0) * data.m)

    return total_loss / n_samples


parser = argparse.ArgumentParser(description='南无阿弥陀佛')
parser.add_argument('--data', type=str, default="traffic", help="the dataset that we want to predict")
parser.add_argument('--model', type=str, default='DSLSTM', help='the model of DSLSTM')
parser.add_argument('--convlstm_units', type=int, default=35, help='number of ConvLSTM hidden units')
parser.add_argument('--lstm_unit', type=int, default=35, help='number of BILSTM hidden unit')
parser.add_argument('--kernel_size', type=int, default= 3, help='size of kernel size')
parser.add_argument('--window', type=int, default=24 * 7, help='window size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers')
parser.add_argument('--horizon', type=int, default=12, help="the horizon that we want to predict. In our paper, "
                                                            "we set it from [3, 6, 12, 24]")
parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--seed', type=int, default=54321, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help="The Index of GPU where we want to run the code")
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()

# location, path, dictionary
path_data_horizon = make_dir(make_dir("results", args.data), args.horizon)
path_log = make_dir(path_data_horizon, "log")
path_model = make_dir(path_data_horizon, "model")
path_result = make_dir(path_data_horizon, "result")

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)

print("----Building models----")
model = eval(args.model).Model(input_dim = Data.m, hidden_dim = args.convlstm_units, output_dim = Data.m,
                               kernel_size = args.kernel_size, lstmhidden_dim=args.lstm_unit,
                               horizon = args.horizon, dropout= args.dropout)

if args.cuda:
    model.cuda()

nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('----number of parameters: %d----' % nParams)
# exit()

if args.L1Loss:
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
best_val = 10000000
best_test_rse = 10000000
optim = Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)

# At any point you can hit Ctrl + C to break out of training early.

writer = SummaryWriter(path_log)

try:
    last_update = 1
    print('----Traning begin----')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
        writer.add_scalar("train_loss", train_loss, epoch)
        val_rse, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                              args.batch_size, "valid")
        writer.add_scalar("val_rse", val_rse, epoch)
        writer.add_scalar("val_rae", val_rae, epoch)
        writer.add_scalar("val_corr", val_corr, epoch)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid corr  {:5.4f} | valid rse {:5.4f} | '
              'valid rae {:5.4f} '.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_corr, val_rse, val_rae))
        # Save the model if the validation loss is the best we've seen so far.

        if val_rse < best_val:
            with open(os.path.join(path_model, "DSLSTM.pkl"), 'wb') as f:
                torch.save(model, f)
            with open(os.path.join(path_log, "log.txt"), "a") as file:
                file.write("epoch:{}, save the model.\r\n".format(epoch))
            last_update = epoch
            best_val = val_rae

        if epoch % 10 == 0:
            test_rse, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                     evaluateL1,
                                                     args.batch_size, "valid")
            print("test corr {:5.4f} | test rse {:5.4f} | test rae {:5.4f} ".format(test_corr, test_rse, test_rae))


        if epoch - last_update == 75:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('----Exiting from training early----')

# Load the best saved model.
print("----Testing begin----")
with open(args.save, 'rb') as f:
    model = torch.load(f)
test_rse, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, "test")
print("test corr {:5.4f} | test rse {:5.4f} | test rae {:5.4f} ".format(test_corr, test_rse, test_rae))