import os
import argparse

import numpy as np

INPUT_FILENAME_PATTERN = 'delay_%d_num_possible_symbols_%d_%s_inputs.npy'
TARGET_FILENAME_PATTERN = 'delay_%d_num_possible_symbols_%d_%s_targets.npy'
DEFAULT_DATA_DIR = os.path.expanduser(os.path.join('~', 'Data', 'CopyProblem'))


def generate_inputs_and_targets(delay, num_examples, num_possible_symbols=10):
  """ Generate copy-problem input, target example sequences.

  Here `delay` must be divisible by 10. This way we can specify the
  number of symbols to be copied so that a majority-vote classifier always
  results in an error rate of 1/12 (as a simple baseline that is then
  independent of `delay`).

  Example 1: `delay = 10`, `num_examples = 1`.
  Possible Inputs:  [3 B B B B B B B B B G B]
  Possible Targets: [B B B B B B B B B B B 3]

  Example 2: `delay = 20`, `num_examples = 1`.
  Possible Inputs:  [1 3 B B B B B B B B B B B B B B B B B B B G B B]
  Possible Targets: [B B B B B B B B B B B B B B B B B B B B B B 1 3]

  Args:
    delay: An integer that's divisible by 10.
    num_examples: An integer.
    num_possible_symbols: An integer.

  Returns:
    A tuple, `(inputs, targets)`. Each is a 2-D int16 array with shape
      `[num_examples, delay/10 + delay + delay/10]`.
  """

  # symbols are 0, 1, ..., num_possible_symbols - 1
  blank_symbol = num_possible_symbols
  go_symbol = num_possible_symbols + 1

  num_symbols_to_copy, remainder = divmod(delay, 10)
  if remainder != 0:
    raise ValueError('delay must be divisible by 10.')

  symbols_to_copy = np.random.randint(0, num_possible_symbols, size=[num_examples, num_symbols_to_copy], dtype=np.int16)
  blank_symbols = blank_symbol * np.ones([num_examples, 1], dtype=np.int16)
  go_symbols = go_symbol * np.ones([num_examples, 1], dtype=np.int16)

  inputs = np.concatenate([symbols_to_copy, np.tile(blank_symbols, [1, delay - 1]),
                           go_symbols, np.tile(blank_symbols, [1, num_symbols_to_copy])],
                          axis=1)

  targets = np.concatenate([np.tile(blank_symbols, [1, num_symbols_to_copy + delay]),
                            symbols_to_copy],
                           axis=1)

  return inputs, targets


def load(data_dir=DEFAULT_DATA_DIR, delay=100, num_possible_symbols=10):
  """ Load Copy Problem data.

  Args:
    data_dir: A string. The data directory.
    delay: An integer.
    num_possible_symbols: An integer.

  Returns:
    A tuple with 6 elements: train inputs, train targets, val inputs, val
    targets, test inputs, test targets. Each is a 2-D int16 array with shape
    `[num_examples, delay/10 + delay + delay/10]`.
  """
  ret = []
  for name in ['train', 'val', 'test']:
    inputs_path = os.path.join(data_dir, INPUT_FILENAME_PATTERN % (delay, num_possible_symbols, name))
    targets_path = os.path.join(data_dir, TARGET_FILENAME_PATTERN % (delay, num_possible_symbols, name))
    if not os.path.exists(inputs_path) or not os.path.exists(targets_path):
      raise ValueError('Data not found in %s. Did you run copyproblem.py?' % data_dir)
    ret.append(np.load(inputs_path))
    ret.append(np.load(targets_path))
  return tuple(ret)


def load_split(data_dir=DEFAULT_DATA_DIR, delay=100, num_possible_symbols=10, val=True):
  """ Load a Copy Problem train, test split.

  Args:
    data_dir: A string. The data directory.
    delay: An integer.
    num_possible_symbols: An integer.
    val: A boolean. If true, return the validation set as the test set.

  Returns:
    A tuple, `(train_inputs, train_targets, test_inputs, test_targets)`.
    Each is a 2-D int16 array with shape
    `[num_examples, delay/10 + delay + delay/10]`.
  """
  outs = load(data_dir, delay, num_possible_symbols)
  train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = outs
  if val:
    test_inputs = val_inputs
    test_targets = val_targets
  return train_inputs, train_targets, test_inputs, test_targets


def main():
  """ Generate and save Copy Problem train, val, and test sets. """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
  parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                      help='''The directory to save data in.''')
  parser.add_argument('--delay', type=int, default=100,
                      help='''The delay between presentation time and copy time. Must be divisible by 10.''')
  parser.add_argument('--num_possible_symbols', type=int, default=10,
                      help='''The number of possible symbols.''')
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
    inputs, targets = generate_inputs_and_targets(args.delay, num_examples[name], args.num_possible_symbols)
    inputs_path = os.path.join(args.data_dir, INPUT_FILENAME_PATTERN % (args.delay, args.num_possible_symbols, name))
    np.save(inputs_path, inputs)
    targets_path = os.path.join(args.data_dir, TARGET_FILENAME_PATTERN % (args.delay, args.num_possible_symbols, name))
    np.save(targets_path, targets)


if __name__ == '__main__':
  main()
