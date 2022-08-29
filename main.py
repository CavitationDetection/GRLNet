'''
Description: 
Author: Yu Sha
Date: 2021-09-28 09:30:05
LastEditors: Yu Sha
LastEditTime: 2022-02-02 14:09:11
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

def train(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # load dataset
    dataset_train = TrainDataset(opts)
    dataset_test = TestDataset(opts)
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, 
                                                    batch_size = opts.batch_size, 
                                                    shuffle = opts.train_batch_shuffle, 
                                                    num_workers = opts.n_threads,
                                                    drop_last = opts.train_drop_last)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                    batch_size = opts.batch_size, 
                                                    shuffle = opts.test_batch_shuffle, 
                                                    num_workers = opts.n_threads,
                                                    drop_last = opts.test_drop_last)

    # get model
    r_g_net = Regional_G_Net().to(device)
    l_g_net = Local_G_Net().to(device)
    d_net = D_Net().to(device) 

    # TODO optimizer
    r_g_net_optimizer = torch.optim.Adam(r_g_net.parameters(), lr = opts.r_g_net_learning_rate)
    l_g_net_optimizer = torch.optim.Adam(l_g_net.parameters(), lr = opts.l_g_net_learning_rate)
    d_net_optimizer = torch.optim.Adam(d_net.parameters(), lr = opts.d_net_learning_rate)

    # Define fake and valid
    fake = torch.ones([opts.batch_size], dtype = torch.float32).to(device)
    valid = torch.zeros([opts.batch_size], dtype = torch.float32).to(device)

    print("Start Training")
    for num_epoch in range(opts.epochs):
        r_g_net.train()
        l_g_net.train()
        d_net.train()
        # for idx, seq in data_loader_train:
        for idx, seq, _ in data_loader_train:
            seq = seq.to(device)
            seq = seq.unsqueeze(1)

            r_g_net_optimizer.zero_grad()
            l_g_net_optimizer.zero_grad()
            d_net_optimizer.zero_grad()

            # noise(Gaussian) added
            input_seq_noise = gaussian(seq,opts.snr)   # Noise
            r_g_net_output = r_g_net(input_seq_noise)
            l_g_net_output = l_g_net(input_seq_noise)

            # d_net output
            d_net_fake_output_r = d_net(r_g_net_output)
            d_net_fake_output_l = d_net(l_g_net_output)

            # d_net real outout
            d_net_real_output = d_net(seq)

            # save image from generator
            # np.save('./results/{epoch}_{idx}_real_samples.npy'.format(epoch = num_epoch, idx = idx), seq.data.cpu().numpy())
            # np.save('./results/{epoch}_{idx}_fake_regional_samples.npy'.format(epoch = num_epoch, idx = idx), r_g_net_output.data.cpu().numpy())
            # np.save('./results/{epoch}_{idx}_fake_local_samples.npy'.format(epoch = num_epoch, idx = idx), l_g_net_output.data.cpu().numpy())
            # np.save('./results/{epoch}_{idx}_noise_samples.npy'.format(epoch = num_epoch, idx = idx), input_seq_noise.data.cpu().numpy())

            # vutils.save_image(seq,'./results/%03d_real_samples_epoch.png' % (num_epoch), nrow = opts.n_row_in_grid, normalize = True)
            # vutils.save_image(r_g_net_output,'./results/%03d_fake_regional_samples_epoch.png' % (num_epoch), nrow = opts.n_row_in_grid, normalize = True)
            # vutils.save_image(l_g_net_output,'./results/%03d_fake_local_samples_epoch.png' % (num_epoch), nrow = opts.n_row_in_grid, normalize = True)
            # vutils.save_image(input_seq_noise,'./results/%03d_noise_samples_epoch.png' % (num_epoch), nrow = opts.n_row_in_grid, normalize = True)

            # d_net fake loss -- measure generator's ability
            d_net_fake_r_loss = F.binary_cross_entropy(d_net_fake_output_r,fake)              
            d_net_fake_l_loss = F.binary_cross_entropy(d_net_fake_output_l,fake)           
            # d_net real loss -- measure generator's ability
            d_net_real_loss = F.binary_cross_entropy(d_net_real_output,valid)             

            # overall loss-- measure discriminator's ability loss
            d_net_sum_loss = opts.alpha * d_net_real_loss + (1 - opts.alpha)*(d_net_fake_r_loss + d_net_fake_l_loss)
            d_net_sum_loss.backward(retain_graph=True)

            r_g_net_optimizer.zero_grad()
            l_g_net_optimizer.zero_grad()

            ##############################################################
            r_g_net_recon_loss = F.mse_loss(r_g_net_output,seq)
            l_g_net_recon_loss = F.mse_loss(l_g_net_output,seq)

            # adversarial loss
            r_g_adversarial_loss = F.binary_cross_entropy(d_net_fake_output_r, valid)             
            l_g_adversarial_loss = F.binary_cross_entropy(d_net_fake_output_l, valid)             

            # sum loss
            g_sum_loss = (1 - opts.adversarial_training_factor)*(r_g_net_recon_loss + l_g_net_recon_loss) + (opts.adversarial_training_factor)*(r_g_adversarial_loss + l_g_adversarial_loss)
            g_sum_loss.backward()
            d_net_optimizer.step()
            r_g_net_optimizer.step()
            l_g_net_optimizer.step()

        r_g_net.eval()
        l_g_net.eval()
        d_net.eval()

        labels = []
        results = []
        test_y = []
        test_y_pred = []
        count = 0
        test_batch_num = 0
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

                test_batch_num += 1

        # print("results",results)
        # print("labels",labels)
        results = np.concatenate(results)                                       
        labels = np.concatenate(labels)
        # print("results",results)
        # print("labels",labels)
        """
        Normal(negative) <===> non cavitation <===> 0 
        Abnormal（positive） <===> cavitation <===> 1
        """
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
        
        print("epoch:{0}, AUC:{1}, ERR:{2}, ERR_thr:{3}, F1-score:{4}, Recall:{5}, Precision:{6}, Accuracy:{7}".format(num_epoch, auc, ERR1, err_thresholds1, f1_score, recall, precision, accuracy))

        if accuracy >= 0.9:
            torch.save(r_g_net.state_dict(), "./results/positive_6_Regional_GNet/model{epoch}_{value}.pth".format(epoch = num_epoch, value = accuracy))
            torch.save(l_g_net.state_dict(), "./results/positive_6_Local_GNet/model{epoch}_{value}.pth".format(epoch = num_epoch, value = accuracy))
            torch.save(d_net.state_dict(), "./results/positive_6_DNet/model{epoch}_{value}.pth".format(epoch = num_epoch, value = accuracy))
        print("Classification Report:\n{}".format(classificationreport))
        print("Confusion Matrix:\n{}".format(cm))
        print("Accuracy:{}".format(accuracy))
        print('Precision:{}'.format(precision))
        print("Recall:{}".format(recall))
        print("F1_score:{}".format(f1_score))
        print("AUC:{}".format(auc))


