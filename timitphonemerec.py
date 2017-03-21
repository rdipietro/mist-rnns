import os
import argparse
import glob
import numpy as np
import scipy.interpolate
import scipy.io.wavfile
import python_speech_features

import utils
import timit

# This is based on Table I and Section II of
# Kai-Fu Lee and Hsiao-Wuen Hon: Speaker-Independent Phone Recognition Using Hidden Markov Models.
# IEEE Transactions on Acoustics, Speech, and Signal Processing. 1989.
FOLDS = {
  'ae':  'ae',     'ah':  'ah',     'ax':  'ah',     'ax-h': 'ah',     'ao':  'ao',
  'aa':  'ao',     'aw':  'aw',     'ay':  'ay',     'b':    'b',      'ch':  'ch',
  'd':   'd',      'dh':  'dh',     'dx':  'dx',     'eh':   'eh',     'el':  'el',
  'l':   'el',     'en':  'en',     'n':   'en',     'nx':   'en',     'er':  'er',
  'axr': 'er',     'ey':  'ey',     'f':   'f',      'g':    'g',      'h#':  'h#',
  'pcl': 'h#',     'tcl': 'h#',     'kcl': 'h#',     'bcl':  'h#',     'dcl': 'h#',
  'gcl': 'h#',     'epi': 'h#',     'pau': 'h#',     'hh':   'hh',     'hv':  'hh',
  'ih':  'ih',     'ix':  'ih',     'iy':  'iy',     'jh':   'jh',     'k':   'k',
  'm':   'm',      'em':  'm',      'ng':  'ng',     'eng':  'ng',     'ow':  'ow',
  'oy':  'oy',     'p':   'p',      'q':   'q',      'r':    'r',      's':   's',
  'sh':  'sh',     'zh':  'sh',     't':   't',      'th':   'th',     'uh':  'uh',
  'uw':  'uw',     'ux':  'uw',     'v':   'v',      'w':    'w',      'y':   'y',
  'z':   'z'
}

TOKEN_VOCAB = sorted(list(set(FOLDS.values())))
assert len(TOKEN_VOCAB) == 40

DEFAULT_DATA_DIR = timit.DEFAULT_DATA_DIR


def _audio_and_labels(prefix):
  """ Load and align a TIMIT wav file with its folded phonemes.

  Returns:
    A tuple, `(audio, labels)`. A 1-D float array and a 1-D int array, both
      with the same shape.
  """

  rate, audio = scipy.io.wavfile.read(prefix + '.wav')
  if rate != timit.SAMPLE_RATE:
    raise RuntimeError('Encountered an unexpected sampling rate of %d in %s' % (rate, prefix))
  audio = np.asarray(audio, dtype=np.float)
  phoneme_data = np.loadtxt(prefix + '.phn', dtype=np.object, comments=None, delimiter=' ',
                            converters={0: int, 1: int, 2: lambda x: str(x, encoding='ascii')})

  n = np.arange(audio.size)
  labels = -1 * np.ones([audio.size], dtype=np.int8)
  for start, end, phoneme in phoneme_data:
    phoneme = FOLDS[phoneme]
    labels[(n >= start) & (n < end)], = utils.tokens_to_ids([phoneme], TOKEN_VOCAB)
  audio_start = np.min(phoneme_data[:, 0])
  audio_end = np.max(phoneme_data[:, 1])
  audio = audio[audio_start:audio_end]
  labels = labels[audio_start:audio_end]
  if any(labels == -1):
    raise RuntimeError('Encountered incomplete labeling in %s' % prefix)

  return audio, labels


def _mfcc_and_labels(audio, labels):
  """ Convert to MFCC features and corresponding (interpolated) labels.

  Returns:
    A tuple, `(mfcc_features, mfcc_labels)`. A 1-D float array and a 1-D int
      array, both with the same shape.
  """
  mfcc_sample_rate = 100.0
  winfunc = lambda x: np.hamming(x)
  mfcc_features = python_speech_features.mfcc(audio, samplerate=timit.SAMPLE_RATE, winlen=0.025,
                                              winstep=1.0/mfcc_sample_rate, lowfreq=85.0,
                                              highfreq=timit.SAMPLE_RATE/2, winfunc=winfunc)
  t_audio = np.linspace(0.0, audio.shape[0] * 1.0 / timit.SAMPLE_RATE, audio.size, endpoint=False)
  t_mfcc = np.linspace(0.0, mfcc_features.shape[0] * 1.0 / mfcc_sample_rate, mfcc_features.shape[0], endpoint=False)
  interp_func = scipy.interpolate.interp1d(t_audio, labels, kind='nearest')
  mfcc_labels = interp_func(t_mfcc)
  return mfcc_features, mfcc_labels


def load(data_dir=DEFAULT_DATA_DIR, mfcc=True):
  """ Load all standardized TIMIT data with folded phoneme labels.

  Args:
    data_dir: A string. The data directory.
    mfcc: A boolean. If True, return MFCC sequences and their corresponding
      label sequences. Otherwise, return raw audio sequences in their
      associated label sequences.

  Returns:
    A tuple with 6 elements: train inputs, train labels, val inputs,
    val labels, test inputs, test labels. Each entry is a list of sequences.
    All input sequences are 2-D float arrays with shape
    `[length, values_per_step]` and all label sequences are 1-D int8 arrays
    with shape `[length]`.
  """
  types = ['mfcc', 'mfcc_labels'] if mfcc else ['audio', 'labels']
  ret = []
  for name in ['train', 'val', 'test']:
    for type in types:
      path = os.path.join(data_dir, name + '_' + type + '.npy')
      if not os.path.exists(path):
        raise ValueError('Data not found in %s. Run timit.py and timitphonemerec.py.' % data_dir)
      data = np.load(path)
      if type == 'audio':
        data = [seq[:, np.newaxis] for seq in data]
      ret.append(data)
  return tuple(ret)


def load_split(data_dir=DEFAULT_DATA_DIR, val=True, mfcc=True, normalize=True):
  """ Load a standardized-TIMIT train, test split.

  Args:
    data_dir: A string. The data directory.
    val: A boolean. If True, return the validation set as the test set.
    mfcc: A boolean. If True, return MFCC sequences and their corresponding
      label Otherwise, return raw audio sequences in their associated
      label sequences.
    normalize: A boolean. If True, normalize each sequence individually by
      centering / scaling.

  Returns:
    A tuple, `(train_inputs, train_labels, test_inputs, test_labels)`. Each is
    a list of sequences. All inputs are 2-D float arrays with shape
    `[length, values_per_step]` and all labels are 1-D int8 arrays with shape
    `[length]`.
  """
  sequence_lists = load(data_dir=data_dir, mfcc=mfcc)
  train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = sequence_lists
  if val:
    test_inputs = val_inputs
    test_labels = val_labels
  if normalize:
    train_inputs = [seq - np.mean(seq, axis=0, keepdims=True) for seq in train_inputs]
    train_inputs = [seq / np.std(seq, axis=0, keepdims=True) for seq in train_inputs]
    test_inputs = [seq - np.mean(seq, axis=0, keepdims=True) for seq in test_inputs]
    test_inputs = [seq / np.std(seq, axis=0, keepdims=True) for seq in test_inputs]
  return train_inputs, train_labels, test_inputs, test_labels


def main():
  """ Further process and simplify standardized TIMIT. """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
  parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                      help='''The standardized-TIMIT data directory.''')
  args = parser.parse_args()

  if not os.path.exists(args.data_dir):
    raise ValueError('%s does not exist. Did you run timit.py?' % args.data_dir)

  for name in ['train', 'val', 'test']:

    print('Processing and saving the %s set..' % name)

    pattern = os.path.join(args.data_dir, name, '*', '*.wav')
    prefixes = [path[:-4] for path in sorted(glob.glob(pattern))]
    audio_label_pairs = [_audio_and_labels(prefix) for prefix in prefixes]
    mfcc_label_pairs = [_mfcc_and_labels(*pair) for pair in audio_label_pairs]

    audio_seqs, label_seqs = zip(*audio_label_pairs)
    np.save(os.path.join(args.data_dir, name + '_audio.npy'), audio_seqs)
    np.save(os.path.join(args.data_dir, name + '_labels.npy'), label_seqs)

    mfcc_seqs, mfcc_label_seqs = zip(*mfcc_label_pairs)
    np.save(os.path.join(args.data_dir, name + '_mfcc.npy'), mfcc_seqs)
    np.save(os.path.join(args.data_dir, name + '_mfcc_labels.npy'), mfcc_label_seqs)


if __name__ == '__main__':
  main()
