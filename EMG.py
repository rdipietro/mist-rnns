import os
#import argparse

import numpy as np
from scipy import io

#INPUT_FILENAME_PATTERN = 'delay_%d_num_possible_symbols_%d_%s_inputs.npy'
#TARGET_FILENAME_PATTERN = 'delay_%d_num_possible_symbols_%d_%s_targets.npy'
#DEFAULT_DATA_DIR = os.path.expanduser(os.path.join('~', 'Data', 'CopyProblem'))
DEFAULT_DATA_DIR = os.path.abspath('EMG_DB')
Idx_sub_train = np.arange(0,15)
Idx_sub_test = np.arange(16,21+1)

Idx_trl_train = np.arange(0,5)
Idx_trl_test = np.arange(6,15)

def load(data_dir):
  """ Load EMG dataset

  Args:
    data_dir: A string. The data directory.


  Returns:
    A tuple with 6 elements: train inputs, train targets, val inputs, val
    targets, test inputs, test targets. Each is a 2-D int16 array with shape
    `[num_examples, delay/10 + delay + delay/10]`.
  """
  
  #mat file 받아오기
  ret = []
  # EMG DB
  mat_EMG = io.loadmat(os.path.join(data_dir,'emg_feat_set_10Hz','EMG_feat_normalized.mat'))
  EMG_feat = mat_EMG['feat']
  #EMG_feat = EMG_feat[:,:3] # RMS특징만 사용
  ret.append(np.concatenate(EMG_feat[1,Idx_trl_train])) # train_inputs 
  ret.append(np.concatenate(EMG_feat[1,Idx_trl_test])) # test_inputs

#  EMG_feat = np.concatenate(np.concatenate(mat_EMG['feat']))
  
  mat_Marker = io.loadmat(os.path.join(data_dir,'DB_markset_10Hz_basecorr_norm_0-1','mark_12.mat'))
  Marker = mat_Marker['marker_set']
 
  ret.append(np.concatenate(Marker[1,Idx_trl_train])) # train_targets
  ret.append(np.concatenate(Marker[1,Idx_trl_test])) # test_targets 



#  Marker = np.concatenate(np.concatenate(mat_Marker['marker_set']))
  

#  ret = []
#  for name in ['train', 'val', 'test']:
#    inputs_path = os.path.join(data_dir, INPUT_FILENAME_PATTERN % (delay, num_possible_symbols, name))
#    targets_path = os.path.join(data_dir, TARGET_FILENAME_PATTERN % (delay, num_possible_symbols, name))
#    if not os.path.exists(inputs_path) or not os.path.exists(targets_path):
#      raise ValueError('Data not found in %s. Did you run copyproblem.py?' % data_dir)
#    ret.append(np.load(inputs_path))
#    ret.append(np.load(targets_path))
  return ret


def load_split(data_dir=DEFAULT_DATA_DIR):
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
  outs = load(data_dir)
  
  train_inputs, test_inputs, train_targets, test_targets = outs
  train_inputs = train_inputs[:,:3]
  test_inputs = test_inputs[:,:3]
  train_targets = train_targets[:,:2]
  test_targets = test_targets[:,:2]

  return train_inputs, train_targets, test_inputs, test_targets


def main():
  """ load EMG mat and get train, val, and test sets. """

#  description = main.__doc__
  load_split(DEFAULT_DATA_DIR)
  
#  formatter_class = argparse.ArgumentDefaultsHelpFormatter
#  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
#  parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
#                      help='''The directory to save data in.''')
#  parser.add_argument('--delay', type=int, default=100,
#                      help='''The delay between presentation time and copy time. Must be divisible by 10.''')
#  parser.add_argument('--num_possible_symbols', type=int, default=10,
#                      help='''The number of possible symbols.''')
#  parser.add_argument('--num_train_examples', type=int, default=100000,
#                      help='''The number of train examples to generate.''')
#  parser.add_argument('--num_val_examples', type=int, default=1000,
#                      help='''The number of validation examples to generate.''')
#  parser.add_argument('--num_test_examples', type=int, default=10000,
#                      help='''The number of test examples to generate.''')
#  args = parser.parse_args()

#  num_examples = {
#    'train': args.num_train_examples,
#    'val': args.num_val_examples,
#    'test': args.num_test_examples
#  }

#  if not os.path.exists(data_dir):
#    os.makedirs(data_dir)

#  for name in ['train', 'val', 'test']:
#    print('Generating %s data ..' % name)
#    inputs, targets = generate_inputs_and_targets(args.delay, num_examples[name], args.num_possible_symbols)
#    inputs_path = os.path.join(args.data_dir, INPUT_FILENAME_PATTERN % (args.delay, args.num_possible_symbols, name))
#    np.save(inputs_path, inputs)
#    targets_path = os.path.join(args.data_dir, TARGET_FILENAME_PATTERN % (args.delay, args.num_possible_symbols, name))
#    np.save(targets_path, targets)


if __name__ == '__main__':
  main()
