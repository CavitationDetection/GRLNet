'''
Description: 
Author: Yu Sha
Date: 2021-09-28 09:30:05
LastEditors: Yu Sha
LastEditTime: 2022-02-15 15:12:37
'''
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import argparse
from train_data_loader import TrainDataset
from test_data_loader import TestDataset
from network import Regional_G_Net,Local_G_Net,D_Net
from sklearn import metrics
from utils import create_dirs
from utils import gaussian
from opts import parse_opts
import warnings
warnings.filterwarnings("ignore")
import os
parser = argparse.ArgumentParser(description='Our model options')
parser.add_argument('--cuda', action = 'store_true', help = 'Choose device to use cpu cuda') 
parser.add_argument('--test_root', action = 'store', type = str,default = '/home/deepthinkers/samson/yusha_workspace/AnomalyDetectionData/Data2017/windowsize/778240_6/TestSpec', 
                        help = 'Root path of test data (normal and abormal data)')
parser.add_argument('--test_label_path', action='store', type=str, default = '/home/deepthinkers/samson/yusha_workspace/AnomalyDetectionData/Data2017/windowsize/778240_6/Label/test_split_label.csv', 
                        help = 'Path of test data label (normal and abnormal data)') 

parser.add_argument('--mess_length', action = 'store', type = int, default = 233472, 
                        help = 'Data length, if data > mess_length, then truncation; else padding(233472,80000)')
parser.add_argument('--batch_size', action='store', type=int, default=4, help='number of data in a batch')
parser.add_argument('--comprehensive_output_factor', action = 'store', type = float, default = 0.5, help = 'DNet output balance factor')
def test(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # load dataset``
    dataset_test = TestDataset(opts)
    # define training and validation data loaders
    data_loader_test = torch.utils.data.DataLoader(dataset_test,shuffle=False,num_workers=1)

    # get model
    l_g_net = Local_G_Net()
    r_g_net = Regional_G_Net()
    d_net = D_Net() 

    l_g_net.load_state_dict(torch.load('results/Local_Generator.pth'))
    r_g_net.load_state_dict(torch.load('results/Regional_Generator.pth'))
    d_net.load_state_dict(torch.load('results/Discirminator.pth'))

    l_g_net.to(device)
    r_g_net.to(device)
    d_net.to(device)


    l_g_net.eval()
    r_g_net.eval()
    d_net.eval()

    labels = []
    results = []
    d = []
    d_regional = []
    d_local = []

    with torch.no_grad():
        for idx, test_seq,test_label in data_loader_test:

            test_seq = test_seq.to(device)
            test_label = test_label.to(device)
            test_seq = test_seq.unsqueeze(1)

            r_g_net_output = r_g_net(test_seq)
            l_g_net_output = l_g_net(test_seq)

            d_net_output_r_g = d_net(r_g_net_output)
            d_net_output_l_g = d_net(l_g_net_output)
            d_net_output = d_net(test_seq)

            comprehensive_d_net_output = opts.comprehensive_output_factor* (d_net_output_r_g + d_net_output_l_g) + (1 - opts.comprehensive_output_factor)* d_net_output

            results.append(comprehensive_d_net_output.cpu().detach().numpy())
            labels.append(test_label.cpu())

            d.append(d_net_output.cpu().detach().numpy())
            d_regional.append(d_net_output_r_g.cpu().detach().numpy())
            d_local.append(d_net_output_l_g.cpu().detach().numpy())

    results = np.concatenate(results)                                       
    labels = np.concatenate(labels)

    d = np.concatenate(d)
    d_regional = np.concatenate(d_regional)
    d_local = np.concatenate(d_local)

    d = pd.DataFrame(d)
    d_regional = pd.DataFrame(d_regional)
    d_local = pd.DataFrame(d_local)

    d.to_csv("./results/d_net.csv")
    d_regional.to_csv("./results/d_regional.csv")
    d_local.to_csv("./results/d_local.csv")

    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels,results,pos_label=1)   
    fnr1 = 1 - tpr1
    err_thresholds1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    ERR1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    d_f1 = np.copy(results)
    d_f1[d_f1 >= err_thresholds1] = 1
    d_f1[d_f1 < err_thresholds1] = 0
        
    # metrics
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    accuracy = metrics.accuracy_score(labels, d_f1)
    precision = metrics.precision_score(labels, d_f1, pos_label=0)
    recall = metrics.recall_score(labels, d_f1, pos_label=0)
    auc = metrics.auc(fpr1, tpr1)
    cm = metrics.confusion_matrix(labels, d_f1)
    classificationreport = metrics.classification_report(labels, d_f1)
        
    print("AUC:{0}, ERR:{1}, ERR_thr:{2}, F1-score:{3}, Recall:{4}, Precision:{5}, Accuracy:{6}".format( auc, ERR1, err_thresholds1, f1_score, recall, precision, accuracy))
    print("Classification Report:\n{}".format(classificationreport))
    print("Confusion Matrix:\n{}".format(cm))
    print("Accuracy:{}".format(accuracy))
    print('Precision:{}'.format(precision))
    print("Recall:{}".format(recall))
    print("F1_score:{}".format(f1_score))
    print("AUC:{}".format(auc))

if __name__ == "__main__":
    opts = parser.parse_args() # Namespace object
    create_dirs('./test')
    test(opts)
