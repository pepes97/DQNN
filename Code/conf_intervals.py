#
# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_data_generator
import evaluation_metrics
import keras_model
from keras.utils import plot_model

import parameter
import utils
import time
from IPython import embed
plot.switch_backend('agg')
from datetime import datetime
import tensorflow as tf
import argparse
from load_json import JSON_Manager
from keras.models import load_model
from complexnn import QuaternionConv2D, QuaternionGRU, QuaternionDense


def collect_test_labels(_data_gen_test, _data_out, classification_mode, quick_test):
    # Collecting ground truth for test data
    nb_batch = 5 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[1]
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa

def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _sed_score, _doa_score):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(311)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.plot(range(nb_epoch), _val_loss, label='val loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(312)
    plot.plot(range(nb_epoch), _sed_score, label='sed_score')
    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
    plot.legend()
    plot.grid(True)

    plot.subplot(313)
    plot.plot(range(nb_epoch), _doa_score, label='doa_score')
    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
    plot.legend()
    plot.grid(True)

    plot.savefig(fig_name)
    plot.close()


def main(args):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
        second input: task_id - (optional) To chose the system configuration in parameters.py. 
                                (default) uses default parameters
    """
    
    # use parameter set defined by user
    task_id = args.params
    params = parameter.get_params(task_id)

    job_id = args.model_name

    model_dir = 'models/'+args.author+'/' if args.author!="" else 'models/'
    utils.create_folder(model_dir)
    unique_name = '{}_ov{}_split{}_{}{}_3d{}_{}'.format(
        params['dataset'], params['overlap'], params['split'], params['mode'], params['weakness'],
        int(params['cnn_3d']), job_id
    )

    model_name = unique_name

    epoch_manager = JSON_Manager(args.author, unique_name)
    logdir = "logs/"+args.author+"/"+ unique_name
    
    unique_name = os.path.join(model_dir, unique_name)
    print("unique_name: {}\n".format(unique_name))

    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only'], shuffle=False
    )

    data_in, data_out = data_gen_test.get_data_sizes()
    print(
        'FEATURES:\n'
        '\tdata_in: {}\n'
        '\tdata_out: {}\n'.format(
            data_in, data_out
        )
    )

    gt = collect_test_labels(data_gen_test, data_out, params['mode'], params['quick_test'])
    sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0])
    doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1])

    print(
        'MODEL:\n'
        '\tdropout_rate: {}\n'
        '\tCNN: nb_cnn_filt: {}, pool_size{}\n'
        '\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'],
            params['nb_cnn3d_filt'] if params['cnn_3d'] else params['nb_cnn2d_filt'], params['pool_size'],
            params['rnn_size'], params['fnn_size']
        )
    )

    model = load_model(os.path.join(model_dir, model_name+"_best_model.h5"), custom_objects={'QuaternionConv2D': QuaternionConv2D,
                                                                                        'QuaternionGRU':QuaternionGRU,
                                                                                        'QuaternionDense': QuaternionDense})
    model.summary()
    plot_model(model, to_file=os.path.join(model_dir, 'model.png'))

    best_metric = epoch_manager.get_best_metric()
    conf_mat = None
    best_conf_mat = epoch_manager.get_best_conf_mat()
    best_epoch = epoch_manager.get_best_epoch()
    patience_cnt = epoch_manager.get_patience_cnt()
    epoch_metric_loss = np.zeros(params['nb_epochs'])
    sed_score=np.zeros(params['nb_epochs'])
    std_score=np.zeros(params['nb_epochs'])
    doa_score=np.zeros(params['nb_epochs'])
    seld_score=np.zeros(params['nb_epochs'])
    tr_loss = np.zeros(params['nb_epochs'])
    val_loss = np.zeros(params['nb_epochs'])
    doa_loss = np.zeros((params['nb_epochs'], 6))
    sed_loss = np.zeros((params['nb_epochs'], 2))
 
    epoch_cnt=0

    pred = model.predict_generator(
        generator=data_gen_test.generate(),
        steps= data_gen_test.get_total_batches_in_data(),
        use_multiprocessing=False,
        verbose=2
    )
    print("pred[1]:",pred[1].shape)
    if params['mode'] == 'regr':
        sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
        print(f"sed_pred: {sed_pred.shape}")
        doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1])
        print(f"doa_pred: {doa_pred.shape}")
        sed_loss[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_test.nb_frames_1s())
        
        
        if params['azi_only']:
            doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred, doa_gt,
                                                                                                sed_pred, sed_gt)
        else:
            doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt,
                                                                                                sed_pred, sed_gt)

        sed_score[epoch_cnt] = np.mean([sed_loss[epoch_cnt, 0], 1-sed_loss[epoch_cnt, 1]])
        print(f"ER: {sed_loss[epoch_cnt, 0]}")
        er = sed_loss[epoch_cnt, 0]
        
        interval = 1.96* np.sqrt( ( (er) * (1- er) )/ sed_pred.shape[0] )
        print(f"interval: {interval}")
        
        doa_score[epoch_cnt] = np.mean([2*np.arcsin(doa_loss[epoch_cnt, 1]/2.0)/np.pi, 1 - (doa_loss[epoch_cnt, 5] / float(doa_gt.shape[0]))])
        seld_score[epoch_cnt] = np.mean([sed_score[epoch_cnt], doa_score[epoch_cnt]])

        doa_error = doa_pred-doa_gt

        doa_error = np.reshape(doa_error, newshape=(doa_error.shape[0],11,2))
        doa_error = doa_error[:, :, 0]
        print(f"doa_error: {doa_error.shape}")
        print(f"doa_error: {doa_error}")
        doa_error = np.reshape(doa_error, newshape=(doa_error.shape[0]*doa_error.shape[1]))
        print(f"doa_error: {doa_error.shape}")
        print(f"doa_error: {doa_error}")

        alpha = 5.0
        # calculate lower percentile (e.g. 2.5)
        lower_p = alpha / 2.0
        # retrieve observation at lower percentile
        lower = max(0.0, np.percentile(doa_error, lower_p))
        print('%.1fth percentile = %.3f' % (lower_p, lower))
        # calculate upper percentile (e.g. 97.5)
        upper_p = (100 - alpha) + (alpha / 2.0)
        # retrieve observation at upper percentile
        upper = min(1.0, np.percentile(doa_error, upper_p))
        print('%.1fth percentile = %.3f' % (upper_p, upper))

        # standard deviation
        std_score[epoch_cnt] = np.std([sed_score[epoch_cnt], doa_score[epoch_cnt]])

        print(f"{er-interval}   /   {er+interval}")
    #lower_confidence, upper_confidence = evaluation_metrics.compute_confidence_interval(best_metric,best_std, params['nb_epochs'], confid_coeff=1.96) # 1.96 for a 95% CI
    
    print("\n----  FINISHED  ----\n")


if __name__ == "__main__":
    try:
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two inputs')
        print('\t>> python seld.py <job-id> --author <name> --params <task-id>')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('-------------------------------------------------------------------------------------------------------')
        
        parser = argparse.ArgumentParser(description='Train a SELDnet')
        parser.add_argument("model_name", type=str, help="Name of the model. If already exists continue the training")
        parser.add_argument('--author', default="", type=str, help="name of the author")
        parser.add_argument('--params', default=1, type=int, help="parameters:\n"
                                                                    "10  ov1 - "
                                                                    "20  ov2 - "
                                                                    "999 quick test")
        args = parser.parse_args()
        print(args)
        sys.exit(main(args))
    except (ValueError, IOError) as e:
        sys.exit(e)
