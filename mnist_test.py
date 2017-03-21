import os
import argparse
import time

import numpy as np
import tensorflow as tf

import models
import utils

import mnist

from mnist_train import INPUT_SIZE, TARGET_SIZE

PARAM_STR_TYPES = [int, str, str, int, float, float, int, int]

TEST_BATCH_SIZE = 100


def parse_args():
  """ Parse command-line arguments and results-dir arguments.

  Returns:
    A tuple, `(args, params_str, layer_kwargs)`.
  """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)

  parser.add_argument('--data_dir', type=str, default=mnist.DEFAULT_DATA_DIR,
                      help='''The directory we will load the data from.''')
  parser.add_argument('--results_dir', type=str, default=mnist.DEFAULT_DATA_DIR,
                      help='''The results directory. Required.''')

  args = parser.parse_args()
  args.data_dir = os.path.expanduser(args.data_dir)
  args.results_dir = os.path.normpath(os.path.expanduser(args.results_dir))

  _, params_str = os.path.split(args.results_dir)
  params = params_str.split('_')
  if len(PARAM_STR_TYPES) != len(params):
    raise ValueError('results_dir is improperly named.')

  params = [type_func(p) for type_func, p in zip(PARAM_STR_TYPES, params)]

  (
    args.permute, args.layer_type, args.activation_type, args.num_hidden_units, args.learning_rate,
    args.optional_bias_shift, args.num_pre_act_mixture_delays, args.trial
  ) = params

  args.pre_act_mixture_delays = 2 ** np.arange(args.num_pre_act_mixture_delays, dtype=np.int)

  layer_kwargs = {
    'optional_bias_shift': args.optional_bias_shift,
    'pre_act_mixture_delays': args.pre_act_mixture_delays,
  }

  return args, params_str, layer_kwargs


def main():
  """ Test an RNN for sequential (possibly permuted) MNIST recognition. """

  args, params_str, layer_kwargs = parse_args()

  outs = mnist.load_split(args.data_dir, val=False, permute=args.permute, normalize=True, seed=0)
  _, _, test_images, test_labels = outs

  # Flatten the images.
  test_inputs = test_images.reshape([len(test_images), -1, INPUT_SIZE])

  # Align sequence-level labels with the appropriate time steps by padding with NaNs,
  # and to do so, first convert the labels to floats.
  length = test_inputs.shape[1]
  pad = lambda x: np.pad(x, [[0, 0], [length - 1, 0], [0, 0]], mode='constant', constant_values=np.nan)
  test_labels = pad(test_labels.reshape([-1, 1, 1]).astype(np.float))

  test_batches = utils.full_bptt_batch_generator(test_inputs, test_labels, TEST_BATCH_SIZE, num_epochs=1,
                                                 shuffle=False)

  model = models.RNNClassificationModel(args.layer_type, INPUT_SIZE, TARGET_SIZE, args.num_hidden_units,
                                        args.activation_type, **layer_kwargs)

  def _error_rate(valid_predictions, valid_targets):
    incorrect_mask = tf.logical_not(tf.equal(tf.argmax(valid_predictions, 1), tf.argmax(valid_targets, 1)))
    return tf.reduce_mean(tf.to_float(incorrect_mask))
  model.error_rate = _error_rate(model.valid_predictions, model.valid_targets)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = False
  sess = tf.Session(config=config)

  saver = tf.train.Saver()
  saver.restore(sess, os.path.join(args.results_dir, 'model.ckpt'))

  error_rates = []
  for batch_inputs, batch_labels in test_batches:

    batch_targets = utils.one_hot(np.squeeze(batch_labels, 2), TARGET_SIZE)
    valid_predictions, valid_targets, batch_error_rates = sess.run(
      [model.valid_predictions, model.valid_targets, model.error_rate],
      feed_dict={model.inputs: batch_inputs,
                 model.targets: batch_targets}
    )
    error_rates.append(batch_error_rates)

  error_rate = np.mean(error_rates, dtype=np.float)
  print('%f' % error_rate)
  with open(os.path.join(args.results_dir, 'test_result.txt'), 'w') as f:
    print('%f' % error_rate, file=f)


if __name__ == '__main__':
    main()
