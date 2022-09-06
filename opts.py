'''
Description: 
Author: Yu Sha
Date: 2021-09-28 09:30:05
LastEditors: Yu Sha
LastEditTime: 2022-05-25 16:52:50
'''
import argparse


# Our model related options
def parse_opts():
    parser = argparse.ArgumentParser(description='Our model options')
    
    parser.add_argument('--cuda', action = 'store_true', help = 'Choose device to use cpu cuda') 

    parser.add_argument('--train_root', action = 'store', type = str,default = 'TestData', 
                        help = 'Root path of training data (normal data)')

    parser.add_argument('--train_label_path', action = 'store', type = str, default = 'Label/test_label.csv', 
                        help = "Path of training data label (normal data, it is only used to load the data without changing the program. see tran_data_loader.py)")

    parser.add_argument('--test_root', action = 'store', type = str,default = 'TestData', 
                        help = 'Root path of test data (normal and abormal data)')

    parser.add_argument('--test_label_path', action='store', type=str, default = 'Label/test_label.csv', 
                        help = 'Path of test data label (normal and abnormal data)') 

    parser.add_argument('--mess_length', action = 'store', type = int, default = 233472, 
                        help = 'Data length, if data > mess_length, then truncation; else padding(233472,80000)')
    
    parser.add_argument('--snr', action = 'store', type = float, default = 2, help = 'Signal-to-noise ratio')

    parser.add_argument('--r_g_net_learning_rate', action = 'store', type = float, default = 0.001, 
                        help = 'Initial learning rate of the High G network')

    parser.add_argument('--l_g_net_learning_rate', action = 'store', type = float, default = 0.001, 
                        help = 'Initial learning rate of the Low G Network')

    parser.add_argument('--d_net_learning_rate', action = 'store', type = float, default = 0.0001, 
                        help = 'Initial learning rate of the Discriminator Network')

    parser.add_argument('--alpha', action = 'store', type = float, default = 0.5, 
                        help = 'Balance factor for training -- fake loss and valid loss')

    parser.add_argument('--adversarial_training_factor', action = 'store', type = float, default = 0.5, 
                        help = 'Loss parameter for generator (reconstruction and adversaria)')
                        
    parser.add_argument('--comprehensive_output_factor', action = 'store', type = float, default = 0.5, 
                        help = 'DNet output balance factor')

    parser.add_argument('--batch_size', action = 'store', type = int, default = 2, 
                        help = 'Batch size')

    parser.add_argument('--epochs', action = 'store', type = int, default = 1, 
                        help = 'train rounds over training set')

    parser.add_argument('--n_threads', action = 'store', type = int, default = 1,
                        help = 'Number of threads for multi-thread loading')

    parser.add_argument('--train_batch_shuffle', action = 'store', type = bool, default = True,
                        help = 'Shuffle input batch for training data')

    parser.add_argument('--test_batch_shuffle', action = 'store', type = bool, default = False,
                        help = 'Shuffle input batch for training data')

    parser.add_argument('--train_drop_last', action = 'store', type = bool, default = True,
                        help = 'Drop the remaining of the batch if the size does not match minimum batch size')

    parser.add_argument('--test_drop_last', action = 'store', type = bool, default = True,
                        help = 'Drop the remaining of the batch if the size does not match minimum batch size')

    args = parser.parse_args()
    return args
