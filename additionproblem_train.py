import os
import sys
import argparse
import shutil
import time

import numpy as np
import tensorflow as tf

import models
import optimizers
import utils

import additionproblem

INPUT_SIZE = 2
TARGET_SIZE = 1
BATCH_SIZE = 100
VAL_BATCH_SIZE = 1000
NUM_OPT_STEPS = 100000
NUM_STEPS_PER_TRAIN_SUMMARY = 10
NUM_STEPS_PER_VAL_SUMMARY = 25


def parse_args():
  """ Parse command-line arguments.

  Returns:
    A tuple, `(args, params_str, layer_kwargs)`.
  """
  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)

  parser.add_argument('--data_dir', type=str, default=additionproblem.DEFAULT_DATA_DIR,
                      help='''The directory we will load data from and save results to.''')
  parser.add_argument('--debug', type=int, default=0,
                      help='''If 1, print some useful information.''')
  parser.add_argument('--length', type=int, default=100,
                      help='''The length of the addition-problem sequences.''')
  parser.add_argument('--layer_type', type=str, default='MISTLayer',
                      help='''The RNN layer to use. See `layers`.''')
  parser.add_argument('--activation_type', type=str, default='tanh',
                      help='''An element-wise activation. See `tensorflow.nn`.''')
  parser.add_argument('--num_hidden_units', type=int, default=139,
                      help='''The number of hidden units to use in the recurrent model.''')
  parser.add_argument('--optimizer', type=str, default='ClippingMomentumOptimizer',
                      help='''The Optimizer to use. See `optimizers`.''')
  parser.add_argument('--learning_rate', type=float, default=1.0,
                      help='''The learning rate.''')
  parser.add_argument('--optional_bias_shift', type=float, default=1.0,
                      help='''Used with LSTMLayer and GRULayer. In the case of LSTM, this is
                              more commonly known as forget-gate bias.''')
  parser.add_argument('--num_pre_act_mixture_delays', type=int, default=8,
                      help='''Used only with the MISTLayer.''')
  parser.add_argument('--trial', type=int, default=0,
                      help='''Useful if we'd like to run multiple trials with identical parameters.''')

  args = parser.parse_args()
  args.data_dir = os.path.expanduser(args.data_dir)
  args.pre_act_mixture_delays = 2 ** np.arange(args.num_pre_act_mixture_delays, dtype=np.int)

  params_str = '_'.join([
    '%d' % args.length,
    '%s' % args.layer_type,
    '%s' % args.activation_type,
    '%04d' % args.num_hidden_units,
    '%.6f' % args.learning_rate,
    '%.1f' % args.optional_bias_shift,
    '%d' % args.num_pre_act_mixture_delays,
    '%02d' % args.trial
  ])

  layer_kwargs = {
    'optional_bias_shift': args.optional_bias_shift,
    'pre_act_mixture_delays': args.pre_act_mixture_delays,
  }

  return args, params_str, layer_kwargs


def main():
  """ Train an RNN for the Addition Problem. """

  args, params_str, layer_kwargs = parse_args()

  save_dir = os.path.join(args.data_dir, 'results', params_str)
  if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
  os.makedirs(save_dir)

  # inputs are [num_examples, length, 2]; targets are [num_examples]. We need
  # everything to be 3-D for the batch generator, and we need to align the
  # targets with the last time step.
  outs = additionproblem.load_split(args.data_dir, args.length, val=True)
  train_inputs, train_targets, val_inputs, val_targets = outs
  pad_width = [[0, 0], [train_inputs.shape[1] - 1, 0], [0, 0]]
  train_targets = train_targets[:, np.newaxis, np.newaxis]
  train_targets = np.pad(train_targets, pad_width, mode='constant', constant_values=np.nan)
  val_targets = val_targets[:, np.newaxis, np.newaxis]
  val_targets = np.pad(val_targets, pad_width, mode='constant', constant_values=np.nan)

  train_batches = utils.full_bptt_batch_generator(train_inputs, train_targets, BATCH_SIZE, shuffle=True)

  model = models.RNNRegressionModel(args.layer_type, INPUT_SIZE, TARGET_SIZE, args.num_hidden_units,
                                    args.activation_type, **layer_kwargs)

  Optimizer = getattr(optimizers, args.optimizer)
  optimizer = Optimizer(args.learning_rate)
  optimize_op = optimizer.minimize(model.valid_stepwise_loss_for_opt)

  tf.summary.scalar('train mse', model.valid_stepwise_loss, collections=['train'])

  model.val_loss = tf.placeholder(tf.float32, shape=[], name='val_loss')
  tf.summary.scalar('val mse', model.val_loss, collections=['val'])

  train_summary_op = tf.summary.merge_all('train')
  val_summary_op = tf.summary.merge_all('val')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = False
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  file_writer = tf.summary.FileWriter(save_dir, graph=sess.graph, flush_secs=10)
  saver = tf.train.Saver()

  best_val_loss = np.inf
  start_time = time.time()
  for step in range(NUM_OPT_STEPS):

    batch_inputs, batch_targets = next(train_batches)

    sess.run(optimize_op,
             feed_dict={model.inputs: batch_inputs,
                        model.targets: batch_targets})

    if step % NUM_STEPS_PER_TRAIN_SUMMARY == 0:

      loss, summary = sess.run([model.valid_stepwise_loss, train_summary_op],
                               feed_dict={model.inputs: batch_inputs,
                                          model.targets: batch_targets})

      file_writer.add_summary(summary, global_step=step)
      with open(os.path.join(save_dir, 'train_status.txt'), 'a') as f:
        line = '%s %06.1f %d %.4f' % (params_str, time.time() - start_time, step, loss)
        print(line, file=f)

    if step % NUM_STEPS_PER_VAL_SUMMARY == 0:

      val_batches = utils.full_bptt_batch_generator(val_inputs, val_targets, VAL_BATCH_SIZE, num_epochs=1,
                                                    shuffle=False)
      losses = []
      for batch_inputs, batch_targets in val_batches:
        valid_predictions, valid_targets, loss = sess.run(
          [model.valid_predictions, model.valid_targets, model.valid_stepwise_loss],
          feed_dict={model.inputs: batch_inputs,
                     model.targets: batch_targets}
        )
        losses.append(loss)

      val_loss = np.mean(losses, dtype=np.float)
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        saver.save(sess, os.path.join(save_dir, 'model.ckpt'))

      summary = sess.run(val_summary_op, feed_dict={model.val_loss: val_loss})
      file_writer.add_summary(summary, global_step=step)
      with open(os.path.join(save_dir, 'val_status.txt'), 'a') as f:
        line = '%s %06.1f %d %.4f %.4f' % (params_str, time.time() - start_time, step,
                                           val_loss, best_val_loss)
        print(line, file=f)
        if args.debug:
          print(line)

  file_writer.close()


if __name__ == '__main__':
  main()
