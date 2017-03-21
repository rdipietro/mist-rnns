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

import mnist

INPUT_SIZE = 1
TARGET_SIZE = 10
BATCH_SIZE = 100
VAL_BATCH_SIZE = 100
NUM_OPT_STEPS = 15000
NUM_STEPS_PER_TRAIN_SUMMARY = 10
NUM_STEPS_PER_VAL_SUMMARY = 150


def parse_args():
  """ Parse command-line arguments.

  Returns:
    A tuple, `(args, params_str, layer_kwargs)`.
  """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)

  parser.add_argument('--data_dir', type=str, default=mnist.DEFAULT_DATA_DIR,
                      help='''The directory we will load data from and save results to.''')
  parser.add_argument('--debug', type=int, default=0,
                      help='''If 1, downsample all images by a factor of 16 and print some useful information.''')
  parser.add_argument('--permute', type=int, default=1,
                      help='''If 1, randomly permute the pixel order of each image (using the same random permutation
                              for every image).''')
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
    '%d' % args.permute,
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
  """ Train an RNN for sequential (possibly permuted) MNIST recognition. """

  args, params_str, layer_kwargs = parse_args()

  save_dir = os.path.join(args.data_dir, 'results', params_str)
  if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
  os.makedirs(save_dir)

  outs = mnist.load_split(args.data_dir, val=True, permute=args.permute, normalize=True, num_val=2000, seed=0)
  train_images, train_labels, val_images, val_labels = outs
  if args.debug:
    train_images = train_images[:, ::4, ::4, :]
    val_images = val_images[:, ::4, ::4, :]

  # Flatten the images.
  train_inputs = train_images.reshape([len(train_images), -1, INPUT_SIZE])
  val_inputs = val_images.reshape([len(val_images), -1, INPUT_SIZE])

  # Align sequence-level labels with the appropriate time steps by padding with NaNs,
  # and to do so, first convert the labels to floats.
  length = train_inputs.shape[1]
  pad = lambda x: np.pad(x, [[0, 0], [length - 1, 0], [0, 0]], mode='constant', constant_values=np.nan)
  train_labels = pad(train_labels.reshape([-1, 1, 1]).astype(np.float))
  val_labels = pad(val_labels.reshape([-1, 1, 1]).astype(np.float))

  train_batches = utils.full_bptt_batch_generator(train_inputs, train_labels, BATCH_SIZE)

  model = models.RNNClassificationModel(args.layer_type, INPUT_SIZE, TARGET_SIZE, args.num_hidden_units,
                                        args.activation_type, **layer_kwargs)

  Optimizer = getattr(optimizers, args.optimizer)
  optimize_op = Optimizer(args.learning_rate).minimize(model.valid_seqwise_loss_for_opt)

  def _error_rate(valid_predictions, valid_targets):
    incorrect_mask = tf.logical_not(tf.equal(tf.argmax(valid_predictions, 1), tf.argmax(valid_targets, 1)))
    return tf.reduce_mean(tf.to_float(incorrect_mask))
  model.error_rate = _error_rate(model.valid_predictions, model.valid_targets)

  tf.summary.scalar('train loss', model.valid_seqwise_loss, collections=['train'])
  tf.summary.scalar('train error rate', model.error_rate, collections=['train'])

  model.val_error_rate = tf.placeholder(tf.float32, name='val_error_rate')
  tf.summary.scalar('val error rate', model.val_error_rate, collections=['val'])

  train_summary_op = tf.summary.merge_all('train')
  val_summary_op = tf.summary.merge_all('val')

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())

  file_writer = tf.summary.FileWriter(save_dir, graph=sess.graph, flush_secs=10)
  saver = tf.train.Saver()

  best_val_error_rate = 1.0
  start_time = time.time()
  for step in range(NUM_OPT_STEPS):

    batch_inputs, batch_labels = next(train_batches)
    batch_targets = utils.one_hot(np.squeeze(batch_labels, 2), TARGET_SIZE)

    sess.run(optimize_op,
             feed_dict={model.inputs: batch_inputs,
                        model.targets: batch_targets})

    if step % NUM_STEPS_PER_TRAIN_SUMMARY == 0:

      error_rate, summary = sess.run([model.error_rate, train_summary_op],
                                     feed_dict={model.inputs: batch_inputs,
                                                model.targets: batch_targets})

      file_writer.add_summary(summary, global_step=step)
      with open(os.path.join(save_dir, 'train_status.txt'), 'a') as f:
        line = '%s %06.1f %d %.4f' % (params_str, time.time() - start_time, step, error_rate)
        print(line, file=f)

    if step % NUM_STEPS_PER_VAL_SUMMARY == 0:

      val_batches = utils.full_bptt_batch_generator(val_inputs, val_labels, VAL_BATCH_SIZE, num_epochs=1,
                                                    shuffle=False)
      error_rates = []
      for i, (batch_inputs, batch_labels) in enumerate(val_batches):
        batch_targets = utils.one_hot(np.squeeze(batch_labels, 2), TARGET_SIZE)
        valid_predictions, valid_targets, batch_error_rates = sess.run(
          [model.valid_predictions, model.valid_targets, model.error_rate],
          feed_dict={model.inputs: batch_inputs,
                     model.targets: batch_targets}
        )
        error_rates.append(batch_error_rates)
        if args.debug and i == 0:
          num_samples = 25
          print('Step: %d. Some targets and predictions:' % step)
          print(np.argmax(valid_targets[:num_samples], axis=1))
          print(np.argmax(valid_predictions[:num_samples], axis=1))

      # This is approximate if the validation batch size doesn't evenly divide
      # the number of validation examples.
      val_error_rate = np.mean(error_rates, dtype=np.float)
      if val_error_rate < best_val_error_rate:
        best_val_error_rate = val_error_rate
        saver.save(sess, os.path.join(save_dir, 'model.ckpt'))

      summary = sess.run(val_summary_op, feed_dict={model.val_error_rate: val_error_rate})
      file_writer.add_summary(summary, global_step=step)
      with open(os.path.join(save_dir, 'val_status.txt'), 'a') as f:
        line = '%s %06.1f %d %.4f %.4f' % (params_str, time.time() - start_time, step,
                                           val_error_rate, best_val_error_rate)
        print(line, file=f)
        if args.debug:
          print(line)

  file_writer.close()


if __name__ == '__main__':
    main()
