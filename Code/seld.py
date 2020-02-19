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


#def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _epoch_metric_loss):
#    plot.figure()
#    nb_epoch = len(_tr_loss)
#    plot.subplot(311)
#    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
#    plot.plot(range(nb_epoch), _val_loss, label='val loss')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(312)
#    plot.plot(range(nb_epoch), _epoch_metric_loss, label='metric')
#    plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
#    plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(313)
#    plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
#    plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
#    plot.legend()
#    plot.grid(True)
#
#    plot.savefig(fig_name)
#    plot.close()
    
#def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _sed_score, _doa_score, epoch_cnt):
#    plot.figure()
#    nb_epoch=epoch_cnt
#   # nb_epoch = len(_tr_loss)
#    plot.subplot(311)
#    #plot.plot(range(nb_epoch), _tr_loss, label='tr loss')
#    #plot.plot(range(nb_epoch), _val_loss, label='val loss')
#    plot.plot(range(nb_epoch), _tr_loss[:nb_epoch], label='tr loss')
#    plot.plot(range(nb_epoch), _val_loss[:nb_epoch], label='val loss')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(312)
#    #plot.plot(range(nb_epoch), _epoch_metric_loss, label='metric')
#    #plot.plot(range(nb_epoch), _sed_loss[:, 0], label='er')
#    #plot.plot(range(nb_epoch), _sed_loss[:, 1], label='f1')
#    plot.plot(range(nb_epoch), _sed_score[:nb_epoch], label='sed_score')
#    plot.plot(range(nb_epoch), _sed_loss[:nb_epoch, 0], label='er')
#    plot.plot(range(nb_epoch), _sed_loss[:nb_epoch, 1], label='f1')
#    plot.legend()
#    plot.grid(True)
#
#    plot.subplot(313)
#    #plot.plot(range(nb_epoch), _doa_loss[:, 1], label='gt_thres')
#    #plot.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_thres')
#    plot.plot(range(nb_epoch), _doa_score, label='doa_score')
#    plot.plot(range(nb_epoch), _doa_loss[:nb_epoch, 1], label='gt_thres')
#    plot.plot(range(nb_epoch), _doa_loss[:nb_epoch, 2], label='pred_thres')
#    plot.legend()
#    plot.grid(True)
#
#    plot.savefig(fig_name)
#    plot.close()

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

    
    session = tf.InteractiveSession()
    
    file_writer = tf.summary.FileWriter(logdir, session.graph)

    data_gen_train = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='train', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only']
    )

    data_gen_test = cls_data_generator.DataGenerator(
        dataset=params['dataset'], ov=params['overlap'], split=params['split'], db=params['db'], nfft=params['nfft'],
        batch_size=params['batch_size'], seq_len=params['sequence_length'], classifier_mode=params['mode'],
        weakness=params['weakness'], datagen_mode='test', cnn3d=params['cnn_3d'], xyz_def_zero=params['xyz_def_zero'],
        azi_only=params['azi_only'], shuffle=False
    )

    data_in, data_out = data_gen_train.get_data_sizes()
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

    model = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                  nb_cnn2d_filt=params['nb_cnn2d_filt'], pool_size=params['pool_size'],
                                  rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                  classification_mode=params['mode'], weights=params['loss_weights'])
    
    initial = epoch_manager.get_epoch()
    if initial!=0:
        print(f"Resume training from epoch {initial}")
        print("Loading already trained model...")
        # In order to load custom layers we need to link the references to the custom objects
        model = load_model(os.path.join(model_dir, model_name+"_model.h5"), custom_objects={'QuaternionConv2D': QuaternionConv2D,
                                                                                        'QuaternionGRU':QuaternionGRU,
                                                                                        'QuaternionDense': QuaternionDense})

    best_metric = epoch_manager.get_best_metric()
    conf_mat = None
    best_conf_mat = epoch_manager.get_best_conf_mat()
    best_epoch = epoch_manager.get_best_epoch()
    patience_cnt = epoch_manager.get_patience_cnt()
    epoch_metric_loss = np.zeros(params['nb_epochs'])
    sed_score=np.zeros(params['nb_epochs'])
    doa_score=np.zeros(params['nb_epochs'])
    seld_score=np.zeros(params['nb_epochs'])
    tr_loss = np.zeros(params['nb_epochs'])
    val_loss = np.zeros(params['nb_epochs'])
    doa_loss = np.zeros((params['nb_epochs'], 6))
    sed_loss = np.zeros((params['nb_epochs'], 2))

    time_hold = tf.placeholder(tf.float32, shape=None, name='time_summary')
    time_summ = tf.summary.scalar('time', time_hold)

    tr_loss_hold = tf.placeholder(tf.float32, shape=None, name='tr_loss_summary')        
    tr_loss_summ = tf.summary.scalar('tr_loss', tr_loss_hold)

    val_loss_hold = tf.placeholder(tf.float32, shape=None, name='val_loss_summary')
    val_loss_summ = tf.summary.scalar('val_loss', val_loss_hold)

    f1_hold = tf.placeholder(tf.float32, shape=None, name='f1_summary')
    f1_summ = tf.summary.scalar('F1_overall', f1_hold)
    
    er_hold = tf.placeholder(tf.float32, shape=None, name='er_summary') 
    er_summ = tf.summary.scalar('ER_overall', er_hold)

    doa_error_gt_hold = tf.placeholder(tf.float32, shape=None, name='doa_error_gt_summary')  
    doa_error_gt_summ = tf.summary.scalar('doa_error_gt', doa_error_gt_hold)

    doa_error_pred_hold = tf.placeholder(tf.float32, shape=None, name='doa_error_pred_summary')
    doa_error_pred_summ = tf.summary.scalar('doa_error_pred', doa_error_pred_hold)
    
    good_pks_hold = tf.placeholder(tf.float32, shape=None, name='good_pks_summary') 
    good_pks_summ = tf.summary.scalar('good_pks_ratio', good_pks_hold)
    
    sed_score_hold = tf.placeholder(tf.float32, shape=None, name='sed_score_summary') 
    sed_score_summ = tf.summary.scalar('sed_score', sed_score_hold)

    doa_score_hold = tf.placeholder(tf.float32, shape=None, name='doa_score_summary') 
    doa_score_summ = tf.summary.scalar('doa_score', doa_score_hold)

    seld_score_hold = tf.placeholder(tf.float32, shape=None, name='seld_score_summary') 
    seld_score_summ = tf.summary.scalar('seld_score', seld_score_hold)

    best_error_metric_hold = tf.placeholder(tf.float32, shape=None, name='best_error_metric_summary') 
    best_error_metric_summ = tf.summary.scalar('best_error_metric', best_error_metric_hold)

    best_epoch_hold = tf.placeholder(tf.float32, shape=None, name='best_epoch_summary') 
    best_epoch_summ = tf.summary.scalar('best_epoch', best_epoch_hold)
    
    merged = tf.summary.merge_all()

    for epoch_cnt in range(initial, params['nb_epochs']):
        start = time.time()
        hist = model.fit_generator(
            generator=data_gen_train.generate(),
            steps_per_epoch=5 if params['quick_test'] else data_gen_train.get_total_batches_in_data(),
            validation_data=data_gen_test.generate(),
            validation_steps=5 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            use_multiprocessing=False,
            epochs=1,
            verbose=1
        )
        tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
        val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

        pred = model.predict_generator(
            generator=data_gen_test.generate(),
            steps=5 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            use_multiprocessing=False,
            verbose=2
        )
        print("pred:",pred[1].shape)
        if params['mode'] == 'regr':
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1])

            sed_loss[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_test.nb_frames_1s())
            if params['azi_only']:
                doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xy(doa_pred, doa_gt,
                                                                                                 sed_pred, sed_gt)
            else:
                doa_loss[epoch_cnt, :], conf_mat = evaluation_metrics.compute_doa_scores_regr_xyz(doa_pred, doa_gt,
                                                                                                  sed_pred, sed_gt)

            sed_score[epoch_cnt] = np.mean([sed_loss[epoch_cnt, 0], 1-sed_loss[epoch_cnt, 1]])
            doa_score[epoch_cnt] = np.mean([2*np.arcsin(doa_loss[epoch_cnt, 1]/2.0)/np.pi, 1 - (doa_loss[epoch_cnt, 5] / float(doa_gt.shape[0]))])
            seld_score[epoch_cnt] = np.mean([sed_score[epoch_cnt], doa_score[epoch_cnt]])

        plot_functions(unique_name, tr_loss, val_loss, sed_loss, doa_loss, sed_score, doa_score)

        patience_cnt += 1
        epoch_manager.increase_patience_cnt()

        model.save('{}_model.h5'.format(unique_name))
        
        if seld_score[epoch_cnt] < best_metric:
            best_metric = seld_score[epoch_cnt]
            epoch_manager.set_best_metric(best_metric)
            
            best_conf_mat = conf_mat
            epoch_manager.set_best_conf_mat(conf_mat)

            best_epoch = epoch_cnt
            epoch_manager.set_best_epoch(best_epoch)

            model.save('{}_best_model.h5'.format(unique_name))
            patience_cnt = 0
            epoch_manager.reset_patience_cnt()
        
        if patience_cnt > params['patience']:
            print(f"\n----  PATIENCE TRIGGERED AFTER {epoch_cnt} EPOCHS  ----\n")
            break

        summary = session.run(merged, feed_dict={time_hold: time.time() - start,
                                                tr_loss_hold: tr_loss[epoch_cnt],
                                                val_loss_hold: val_loss[epoch_cnt],
                                                f1_hold: sed_loss[epoch_cnt, 1],
                                                er_hold: sed_loss[epoch_cnt, 0],
                                                doa_error_gt_hold: doa_loss[epoch_cnt, 1],
                                                doa_error_pred_hold: doa_loss[epoch_cnt, 2],
                                                good_pks_hold: doa_loss[epoch_cnt, 5] / float(sed_gt.shape[0]),
                                                sed_score_hold: sed_score[epoch_cnt],
                                                doa_score_hold: doa_score[epoch_cnt],
                                                seld_score_hold: seld_score[epoch_cnt],
                                                best_error_metric_hold: best_metric,
                                                best_epoch_hold: best_epoch})
        file_writer.add_summary(summary, epoch_cnt)

        print('epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
            'F1_overall: %.2f, ER_overall: %.2f, '
            'doa_error_gt: %.2f, doa_error_pred: %.2f, good_pks_ratio:%.2f, '
            'sed_score: %.2f, doa_score: %.2f, best_error_metric: %.2f, best_epoch : %d' %
            (
                epoch_cnt, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                sed_loss[epoch_cnt, 1], sed_loss[epoch_cnt, 0],
                doa_loss[epoch_cnt, 1], doa_loss[epoch_cnt, 2], doa_loss[epoch_cnt, 5] / float(sed_gt.shape[0]),
                sed_score[epoch_cnt], doa_score[epoch_cnt], best_metric, best_epoch
            )
        )
        epoch_manager.increase_epoch()
    
    print("\n----  FINISHED TRAINING  ----\n")

    print('best_conf_mat : {}'.format(best_conf_mat))
    print('best_conf_mat_diag : {}'.format(np.diag(best_conf_mat)))
    print('saved model for the best_epoch: {} with best_metric: {},  '.format(best_epoch, best_metric))
    print('DOA Metrics: doa_loss_gt: {}, doa_loss_pred: {}, good_pks_ratio: {}'.format(
        doa_loss[best_epoch, 1], doa_loss[best_epoch, 2], doa_loss[best_epoch, 5] / float(sed_gt.shape[0])))
    print('SED Metrics: ER_overall: {}, F1_overall: {}'.format(sed_loss[best_epoch, 0], sed_loss[best_epoch, 1]))
    print('unique_name: {} '.format(unique_name))


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
