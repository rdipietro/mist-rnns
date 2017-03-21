import os
import argparse

import numpy as np

INPUT_FILENAME_PATTERN = 'length_%d_%s_inputs.npy'
TARGET_FILENAME_PATTERN = 'length_%d_%s_targets.npy'
DEFAULT_DATA_DIR = os.path.expanduser(os.path.join('~', 'Data', 'AdditionProblem'))


def generate_inputs_and_targets(length, num_examples):
  """ Generate addition-problem input, target example sequences.

  Example input sequence (transposed): `[[0.1 0.9 0.7 0.7 0.2],
                                         [0.0 1.0 0.0 0.0 1.0]]`
  Corresponding target: `1.1`

  Note that we follow Arjovsky et al., Unitary Evolution Recurrent Neural
  Networks, 2016 and mark one number randomly from the left half of the
  sequence and one randomly from the right half of the sequence.

  Args:
    length: An integer.
    num_examples: An integer.

  Returns:
    A tuple, `(inputs, targets)`. `inputs` is a 3-D float array with shape
    `[num_examples, length, 2]`. `targets` is a 1-D float array with shape
    `[num_examples]`.
  """
  inputs_1 = np.random.uniform(size=[num_examples, length])
  inputs_2 = np.zeros([num_examples, length], dtype=np.float)
  inputs_2[:, 0] = 1.0
  inputs_2[:, -1] = 1.0
  half_of_length, _ = divmod(length, 2)
  for row in inputs_2:
    np.random.shuffle(row[:half_of_length])
    np.random.shuffle(row[half_of_length:])
  inputs = np.concatenate([inputs_1[:, :, np.newaxis], inputs_2[:, :, np.newaxis]], axis=2)
  targets = np.sum(inputs_1 * inputs_2, axis=1)
  return inputs, targets


def load(data_dir=DEFAULT_DATA_DIR, length=100):
  """ Load Addition Problem data.

  Args:
    data_dir: A string. The data directory.
    length: An integer.

  Returns:
    A tuple with 6 elements: train inputs, train targets, val inputs, val
    targets, test inputs, test targets. All inputs are 3-D float arrays with
    shape `[num_examples, length, 2]`. All targets are 1-D float arrays with
    shape `[num_examples]`.
  """
  ret = []
  for name in ['train', 'val', 'test']:
    inputs_path = os.path.join(data_dir, INPUT_FILENAME_PATTERN % (length, name))
    targets_path = os.path.join(data_dir, TARGET_FILENAME_PATTERN % (length, name))
    if not os.path.exists(inputs_path) or not os.path.exists(targets_path):
      raise ValueError('Data not found in %s. Did you run additionproblem.py?' % data_dir)
    ret.append(np.load(inputs_path))
    ret.append(np.load(targets_path))
  return tuple(ret)


def load_split(data_dir=DEFAULT_DATA_DIR, length=100, val=True):
  """ Load an Addition Problem train, test split.

  Args:
    data_dir: A string. The data directory.
    length: An integer.
    val: A boolean. If true, return the validation set as the test set.

  Returns:
    A tuple, `(train_inputs, train_targets, test_inputs, test_targets)`.
    All inputs are 3-D float arrays with shape `[num_examples, length, 2]`.
    All targets are 1-D float arrays with shape `[num_examples]`.
  """
  outs = load(data_dir, length)
  train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = outs
  if val:
    test_inputs = val_inputs
    test_targets = val_targets
  return train_inputs, train_targets, test_inputs, test_targets


def main():
  """ Generate and save Addition Problem train, val, and test sets. """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
  parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                      help='''The directory to save data in.''')
  parser.add_argument('--length', type=int, default=100,
                      help='''The length of all example sequences.''')
  parser.add_argument('--num_train_examples', type=int, default=100000,
                      help='''The number of train examples to generate.''')
  parser.add_argument('--num_val_examples', type=int, default=1000,
                      help='''The number of validation examples to generate.''')
  parser.add_argument('--num_test_examples', type=int, default=10000,
                      help='''The number of test examples to generate.''')
  args = parser.parse_args()

  num_examples = {
    'train': args.num_train_examples,
    'val': args.num_val_examples,
    'test': args.num_test_examples
  }

  if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

  for name in ['train', 'val', 'test']:
    print('Generating %s data ..' % name)
    inputs, targets = generate_inputs_and_targets(args.length, num_examples[name])
    inputs_path = os.path.join(args.data_dir, INPUT_FILENAME_PATTERN % (args.length, name))
    np.save(inputs_path, inputs)
    targets_path = os.path.join(args.data_dir, TARGET_FILENAME_PATTERN % (args.length, name))
    np.save(targets_path, targets)


if __name__ == '__main__':
  main()
