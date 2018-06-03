import torch
import torch.autograd as autograd
import matplotlib
matplotlib.use('agg')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import VLSTM
import pandas as pd
import numpy as np
import sys
import utils
import os
import random
from tensorboardX import SummaryWriter
from Data_loader_vanilla import DataLoader
import math
import argparse
import pickle 
from tqdm import tqdm


# Defines program args
parser = argparse.ArgumentParser()
# RNN size parameter (dimension of the output/hidden state)
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size')
parser.add_argument('--nb_iter', type=int, default=25,
                    help='Nb epochs')
parser.add_argument('--embedded_input', type=int, default=64,
                    help='size of spatial embedding')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--loading_checkpoint', type=str, default='0',
                    help='Checkpoint model')
parser.add_argument('--plot_save', type=str, default='vlstm',
                    help='name for checkpoint folders')
parser.add_argument('--data_path', type=str, default='../preprocessed_data/',
                    help='Data location')
parser.add_argument('--spec_csv_path', type=str, default='specs.csv',
                    help='Data location')
parser.add_argument('--scaling_factor', type=int, default=10,
                    help='Scaling factor for coordinates')
parser.add_argument('--test_train_folder', type=str, default=None,
                    help='folder containing the file names that will be used for test and training')
parser.add_argument('--use_positions', action='store_true',
                    help='Train with velocities instead of positions')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Learning rate')
parser.add_argument('--eval_folder', type=str,
                    default=None, help='folder to save plots')


def save_imgs(batch, img_path,idx, n_neighbors=5, xlim=[-150, 150], ylim=[-150, 150]):
    global obs,use_speeds,batch_size
    if use_speeds:
        batch_for_network = torch.cat(batch[1], 0).transpose(0, 1)
    else:
        batch_for_network = torch.cat(batch[0], 0).transpose(0, 1)

    batch_for_network = Variable(batch_for_network.squeeze(0))
    # Forward through network
    first_pos = torch.cat(batch[0], 0)[:, 7, :][:, [2, 3]]
    results, speeds = net.forward(
            batch_for_network[:, :, [2, 3]][:obs, :, :])

    if use_speeds:
        first_pos = torch.cat(batch[0], 0)[:, 7, :][:, [2, 3]]
        speeds[0, :, :] = torch.add(first_pos, speeds[0, :, :])
        pts_predicted = torch.cumsum(speeds, 0)
    else:
        pts_predicted = speeds
    idx__ = batch_size*idx
    for x in range(len(batch[1])):
        try:
            utils.plot_trajectory(batch[0][x].squeeze(0), pts_predicted[:, x, :],
                                          batch[2][x], n_neighbors,
                                          img_path+'_'+str(idx__+x),xlim = xlim,ylim= ylim)
        except:
            continue
    return batch[0], pts_predicted, batch[2]





def iterate_and_save(data, saving_path):
    saved_arr = []
    for idx, batch in tqdm(enumerate(data)):
        saved_arr.append(save_imgs(batch, saving_path,idx))

    with open(saving_path + '.pkl', 'wb') as fo:
        pickle.dump(saved_arr, fo)

def test_function(model, test_data, save_plot, epoch, it, save_dir=('imgs_vlstm_nr/', 'checkpoints_vlstm_nr/'), num=12, plot_loss=False, comment=''):
    global writer, use_speeds
    # Variables
    saved_for_plots = []
    obs = 8
    loss_display_test, rmse_display, l2_display, final_l2_display = 0, 0, 0, 0

    # Sample plots to save
    saved_plots = random.sample(range(len(test_data)), 3)
    saved_plots = dict([(int(math.floor(x / test_data.batch_size)), x %
                         test_data.batch_size) for x in saved_plots])

    for idx, batch in enumerate(test_data):
        batch_for_network = None
        if use_speeds:
            batch_for_network = Variable(
                torch.cat(batch[1], 0).transpose(0, 1))
        else:
            batch_for_network = Variable(
                torch.cat(batch[0], 0).transpose(0, 1))

        # Forward through network
        results, speeds = net.forward(
            batch_for_network[:, :, [2, 3]][:obs, :, :])

        # Compute loss
        loss = utils.get_lossfunc(results[:, :, 0], results[:, :, 1], results[:, :, 2],
                                  results[:, :, 3], results[:, :, 4], batch_for_network[8:, :, 2], batch_for_network[8:, :, 3])
        pts_predicted = None
        if use_speeds:
            first_pos = torch.cat(batch[0], 0)[:, 7, :][:, [2, 3]]
            speeds[0, :, :] = torch.add(first_pos, speeds[0, :, :])
            pts_predicted = torch.cumsum(speeds, 0)
        else:
            pts_predicted = speeds
        # Compute accuracies
        tmp_true_pos = torch.cat(batch[0], 0).transpose(0, 1)
        acc = utils.get_avg_displacement(
            tmp_true_pos[8:, :, :][:, :, [2, 3]], pts_predicted)
        acc_l2, acc_final = utils.get_accuracy(
            tmp_true_pos[8:, :, :][:, :, [2, 3]], pts_predicted)

        if len(saved_for_plots) < 50 and random.choice([True, False]):
            saved_for_plots.append((batch, pts_predicted))

        loss_display_test += loss.data[0]
        acc = (torch.sum(acc))
        acc_l2 = torch.sum(acc_l2)
        rmse_display += acc
        l2_display += acc_l2
        final_l2_display += torch.sum(acc_final)

        if (idx in saved_plots) and save_plot:
            plot_true, plot_pred = batch, pts_predicted
            x = saved_plots[idx]
            utils.plot_trajectory(batch[0][x].squeeze(0), plot_pred[:, x, :], batch[2][
                                  x], 5, "{}vanilla_lstm_it_{}_i_{}_b_{}_test_{}".format(save_dir[0], epoch, idx, x, comment))

    if save_plot:
        utils.save_checkpoint({
            'true_pred': saved_for_plots,
        }, save_dir[1] + 'test_data_plot_o_lstm_{}_{}.pth.tar'.format(epoch, comment))

    print('[Loss_Test_{}] {}'.format(
        comment, loss_display_test / len(test_data)))
    print('[Accuracy_mse_Test_{}] {}'.format(comment,
                                             rmse_display / len(test_data)))
    print('[Accuracy_l2_Test_{}] {}'.format(
        comment, l2_display / len(test_data)))
    print('[Accuracy_final_Test_{}] {}'.format(comment,
                                               final_l2_display / len(test_data)))

    print('-------------------------------------------------------------')
    if (plot_loss) and (writer is not None):
        writer.add_scalar('Loss Test {}'.format(comment),
                          (loss_display_test / len(test_data)), it)
        writer.add_scalar('acc_l2 Test {}'.format(comment),
                          (l2_display / len(test_data)), it)


# Define Model params
args_input = parser.parse_args()
print(args_input.use_positions)
batch_size = args_input.batch_size
nb_iter = args_input.nb_iter
embedded_input = args_input.embedded_input
hidden_size = args_input.hidden_size
loading_checkpoint = args_input.loading_checkpoint
print(loading_checkpoint)
save_dir = ('vlstm_imgs_{}/'.format(args_input.plot_save),
            'vlstm_checkpoints_{}/'.format(args_input.plot_save))
for directory in save_dir:
    if not os.path.exists(directory):
        os.makedirs(directory)
use_speeds = not (args_input.use_positions)
# Instanciate model and data loader
args = {'embedded_input': embedded_input, 'hidden_size': hidden_size}
net = VLSTM(args)
net = utils.en_cuda(net)
it = 0
min_epoch = 0
optimizer = torch.optim.RMSprop(net.parameters(), lr=args_input.lr)

# Model loading
if loading_checkpoint != '0':
    load_params = torch.load(loading_checkpoint)
    net.load_state_dict(load_params['state_dict'])
    optimizer.load_state_dict(load_params['optimizer'])
    min_epoch = load_params['epoch']
    it = load_params['iteration']
    print('model loaded :' + loading_checkpoint)

# Load data
# Select adequate samples
if args_input.test_train_folder is None:
    # Load data
    df_specs = pd.read_csv(args_input.spec_csv_path)
    df_specs = df_specs[df_specs.Dataset != 'stanford']
    files_names = df_specs['Dataset'] + '/' + df_specs['File'] + \
        '_' + df_specs['Track_ID'].astype(int).astype(str)
    filters = [df_specs.is_static, ~df_specs.is_static & (
        df_specs.Mean_rotation > 30) & (df_specs.nb_critical_neighbors > 5), ~df_specs.is_static & (
        df_specs.Mean_rotation < 30) & (df_specs.nb_critical_neighbors < 5),
    ]
    sample_static_test = files_names[filters[0]].sample(100)
    sample_crowded_test = files_names[filters[1]].sample(100)
    sample_non_crowded_test = files_names[filters[2]].sample(100)
    list_train_pts = files_names[~(files_names.isin(sample_static_test) | files_names.isin(
        sample_crowded_test) | files_names.isin(sample_non_crowded_test))]
    if not os.path.exists('test_train_files/'):
        os.makedirs('test_train_files/')

    sample_static_test.to_csv(
        'test_train_files/static_test.csv')
    sample_crowded_test.to_csv(
        'test_train_files/crowded_test.csv')
    sample_non_crowded_test.to_csv(
        'test_train_files/non_crowded_test.csv')
    list_train_pts.to_csv('test_train_files/train.csv')
    del filters
    del df_specs
    del files_names
else:
    sample_static_test = pd.read_csv(
        '{}/static_test.csv'.format(args_input.test_train_folder), header=None)[1]
    sample_crowded_test = pd.read_csv(
        '{}/crowded_test.csv'.format(args_input.test_train_folder), header=None)[1]
    sample_non_crowded_test = pd.read_csv(
        '{}/non_crowded_test.csv'.format(args_input.test_train_folder), header=None)[1]
    list_train_pts = pd.read_csv(
        '{}/train.csv'.format(args_input.test_train_folder), header=None)[1]


sample_static_test = DataLoader(batch_size, sample_static_test, args_input.data_path,
                                scaling_factor=args_input.scaling_factor)
sample_crowded_test = DataLoader(batch_size, sample_crowded_test, args_input.data_path,
                                 scaling_factor=args_input.scaling_factor)
sample_non_crowded_test = DataLoader(batch_size, sample_non_crowded_test, args_input.data_path,
                                     scaling_factor=args_input.scaling_factor)

print('Test data loaded: {} samples'.format(len(sample_static_test) +
                                            len(sample_crowded_test) + len(sample_non_crowded_test)))

train_data = DataLoader(batch_size, list_train_pts, args_input.data_path,
                        scaling_factor=args_input.scaling_factor)
print('Data Loaded : {} samples'.format(len(train_data)))


num = 12
obs = 8
if args_input.eval_folder is None:
    loss_display_mid, acc_display_mid = 0, 0
    cnt = 0
    writer = SummaryWriter(comment='vlstm_' + args_input.plot_save)
    for epoch in range(min_epoch, nb_iter):
        # Trajectories to plot
        saved_for_plots = []
        saved_plots = random.sample(range(len(train_data)), 1)
        saved_plots = [(int(math.floor(x / train_data.batch_size)), x %
                        train_data.batch_size) for x in saved_plots]
        plot_pred, plot_true = None, None
        loss_display, rmse_display, l2_display, final_l2_display = 0, 0, 0, 0
        for idx, batch in enumerate(train_data):
            optimizer.zero_grad()
            net.zero_grad()

            # Count number of samples that we iterated on
            cnt += len(batch[0])
            batch_for_network = None
            if use_speeds:
                batch_for_network = torch.cat(batch[1], 0).transpose(0, 1)
            else:
                batch_for_network = torch.cat(batch[0], 0).transpose(0, 1)

            batch_for_network = Variable(batch_for_network)
            # Forward through network
            results, speeds = net.forward(
                batch_for_network[:, :, [2, 3]][:obs, :, :])

            # Compute loss
            loss = utils.get_lossfunc(results[:, :, 0], results[:, :, 1], results[:, :, 2], results[:, :, 3],
                                      results[:, :, 4], batch_for_network[8:, :, 2], batch_for_network[8:, :, 3])
            pts_predicted = None

            if use_speeds:
                first_pos = torch.cat(batch[0], 0)[:, 7, :][:, [2, 3]]
                speeds[0, :, :] = torch.add(first_pos, speeds[0, :, :])
                pts_predicted = torch.cumsum(speeds, 0)
            else:
                pts_predicted = speeds

            # Compute accuracies
            tmp_true_pos = torch.cat(batch[0], 0).transpose(0, 1)
            acc = utils.get_avg_displacement(
                tmp_true_pos[8:, :, :][:, :, [2, 3]], pts_predicted)
            acc_l2, acc_final = utils.get_accuracy(
                tmp_true_pos[8:, :, :][:, :, [2, 3]], pts_predicted)

            if len(saved_for_plots) < 50 and random.choice([True, False]):
                saved_for_plots.append((batch, pts_predicted))

            loss_display += loss.data[0]
            acc = (torch.sum(acc))
            acc_l2 = torch.sum(acc_l2)
            rmse_display += acc
            l2_display += acc_l2
            final_l2_display += torch.sum(acc_final)
            loss_display_mid += loss.data[0]
            acc_display_mid += acc_l2
            loss.backward()
            optimizer.step()
            if (idx == saved_plots[0][0]) and (epoch % 2 == 0):
                plot_true, plot_pred = batch, pts_predicted
                x = saved_plots[0][1]
                utils.plot_trajectory(batch[0][x].squeeze(0), plot_pred[:, x, :], batch[2][x],
                                      5, "{}/vanilla_lstm_it_{}_i_{}_b_{}".format(save_dir[0], args_input.plot_save, epoch, saved_plots[0][0], x))
            it += 1
            if(it % 2 == 0):
                writer.add_scalar('Loss', loss.data[0] / len(batch[0]), it)
                writer.add_scalar('Accuracy', acc / len(batch[0]), it)
            if(cnt > 500):
                writer.add_scalar('Loss_epoch', loss_display_mid / cnt, it)
                writer.add_scalar('Accuracy_l2_epoch', acc_display_mid / cnt, it)
                loss_display_mid, acc_display_mid, cnt = 0, 0, 0

        print('iteration {}'.format(epoch))
        print('[Loss] {}'.format(loss_display / len(train_data)))
        print('[Accuracy_mse] {}'.format(rmse_display / len(train_data)))
        print('[Accuracy_l2] {}'.format(l2_display / len(train_data)))
        print('[Accuracy_final] {}'.format(final_l2_display / len(train_data)))
        print('-------------------------------------------------------------')
        test_function(net, sample_crowded_test, (epoch % 2 == 0),
                      epoch, it, save_dir=save_dir, comment='crowded', plot_loss=True)
        test_function(net, sample_non_crowded_test, (epoch % 2 == 0),
                      epoch, it, save_dir=save_dir, comment='non_crowded')
        test_function(net, sample_static_test, (epoch % 2 == 0),
                      epoch, it, save_dir=save_dir, comment='static')

        print()
        if epoch % 2 == 0:
            utils.save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': it + 1,
                'epoch': epoch + 1,
            }, save_dir[1] + 'checkpoint_vanilla_lstm_{}.pth.tar'.format(epoch))
            utils.save_checkpoint({
                'true_pred': saved_for_plots,
            }, save_dir[1] + 'data_plot_vanilla_lstm_{}.pth.tar'.format(epoch))
            print('Model saved')
            print()
    writer.close()
else:
    if not os.path.isdir(args_input.eval_folder):
        os.mkdir(args_input.eval_folder)
    iterate_and_save(
        train_data, '{}/o_lstm_train'.format(args_input.eval_folder))
    iterate_and_save(sample_crowded_test,
                     '{}/o_lstm_crowded_test'.format(args_input.eval_folder))
    iterate_and_save(sample_non_crowded_test,
                     '{}/o_lstm_non_crowded_test'.format(args_input.eval_folder))
    iterate_and_save(sample_static_test,
                     '{}/o_lstm_static'.format(args_input.eval_folder))
    print('DONE')