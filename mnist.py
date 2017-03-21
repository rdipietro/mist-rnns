import os
import argparse
import urllib.request
import gzip

import numpy as np

URL_BASE = 'http://yann.lecun.com/exdb/mnist/'
FILENAMES = {
  'train_images': 'train-images-idx3-ubyte.gz',
  'train_labels': 'train-labels-idx1-ubyte.gz',
  'test_images': 't10k-images-idx3-ubyte.gz',
  'test_labels': 't10k-labels-idx1-ubyte.gz'
}

HEIGHT = 28
WIDTH = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

DEFAULT_DATA_DIR = os.path.expanduser(os.path.join('~', 'Data', 'MNIST'))


def _load_mnist_images_gzip(path):
  """
  Returns:
      A float NumPy array with values between 0.0 and 1.0 and shape
      `[num_images, HEIGHT, WIDTH, NUM_CHANNELS]`.
  """

  with gzip.open(path, 'rb') as f:
    raw_contents = f.read()

  raw_array = np.fromstring(raw_contents, dtype=np.uint8)

  # Discard metadata: the magic number, the number of images,
  # the number of rows, and the number of columns, each as a 32-bit int,
  # totaling 16 bytes. By discarding these, endianness will never be an issue.
  raw_array = raw_array[16:]

  array = raw_array.reshape(-1, HEIGHT, WIDTH, NUM_CHANNELS).astype(np.float) / 255.0
  return array


def _load_mnist_labels_gzip(path):
  """
  Returns:
    An int NumPy array with shape `[num_images]`.
  """

  with gzip.open(path, 'rb') as f:
    raw_contents = f.read()

  raw_array = np.fromstring(raw_contents, dtype=np.uint8)

  # Discard metadata: the magic number and the number of labels, each as a
  # 32-bit int. By discarding these, endianness will never be an issue.
  raw_array = raw_array[8:]

  array = raw_array.astype(np.int)
  return array


def load(data_dir=DEFAULT_DATA_DIR):
  """ Load MNIST.

  Args:
    data_dir: A string. The data directory.

  Returns:
    A tuple, (train_images, train_labels, test_images, test_labels), each a
    NumPy array. The image arrays have type float and shape
    `[num_images, HEIGHT, WIDTH, 1]`, with values in `[0.0, 1.0]`.
    The label arrays have type int and shape `[num_images]`, with values
    in `[0, NUM_CLASSES-1]`.
  """

  paths = {name: os.path.join(data_dir, filename)
           for name, filename in FILENAMES.items()}
  if any(not os.path.exists(path) for path in paths.values()):
    raise ValueError("Data not found in %s. Run mnist.py to download the data." % data_dir)

  train_images = _load_mnist_images_gzip(paths['train_images'])
  train_labels = _load_mnist_labels_gzip(paths['train_labels'])
  test_images = _load_mnist_images_gzip(paths['test_images'])
  test_labels = _load_mnist_labels_gzip(paths['test_labels'])

  return train_images, train_labels, test_images, test_labels


def load_split(data_dir=DEFAULT_DATA_DIR, val=True, permute=False, normalize=True, num_val=2000, seed=None):
  """ Load an MNIST train, test split.

  Args:
    data_dir: A string. The data directory.
    val: A boolean. If True, do not return any real test data. Instead, return
      a test set that is actually a validation set, formed by splitting the
      train set into two parts.
    permute: A boolean. If True, permute pixels in a random (but fixed) order.
    normalize: A boolean. If True, normalize each input sequence individually
      by centering / scaling.
    num_val: Only used if `val` is True. The number of training examples to use
      for validation.
    seed: An integer or None. Can be specified if we're permuting and would like
      reproducibility.

  Returns:
    A tuple, `(train_images, train_labels, test_images, test_labels)`.
    `train_images` and `test_images` are float NumPy arrays with shape
    `[num_images, HEIGHT, WIDTH, 1]` and `train_labels` and `test_labels` are
    1-D int NumPy arrays with shape `[num_images]`.
  """

  train_images, train_labels, test_images, test_labels = load(data_dir)
  H, W, C = HEIGHT, WIDTH, NUM_CHANNELS

  if val:
    num_train = train_images.shape[0] - num_val
    train_images, test_images = np.split(train_images, [num_train])
    train_labels, test_labels = np.split(train_labels, [num_train])

  num_train, num_test = train_images.shape[0], test_images.shape[0]

  if permute:
    prng = np.random.RandomState(seed=seed)
    permutation_ind = prng.permutation(H * W)
    train_images = train_images.reshape([num_train, -1])
    train_images = train_images[:, permutation_ind]
    train_images = train_images.reshape([num_train, H, W, C])
    test_images = test_images.reshape([num_test, -1])
    test_images = test_images[:, permutation_ind]
    test_images = test_images.reshape([num_test, H, W, C])

  if normalize:
    train_images -= np.mean(train_images, axis=(1, 2), keepdims=True)
    train_images /= np.std(train_images, axis=(1, 2), keepdims=True)
    test_images -= np.mean(test_images, axis=(1, 2), keepdims=True)
    test_images /= np.std(test_images, axis=(1, 2), keepdims=True)

  return train_images, train_labels, test_images, test_labels


def main():
  """ Download MNIST. """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
  parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                      help='''The directory in which the MNIST data will be saved.''')
  args = parser.parse_args()

  if os.path.exists(args.data_dir):
    raise ValueError('%s already exists. ' % args.data_dir + 'If you intend to overwrite it, delete it first.')

  os.makedirs(args.data_dir)
  for name, filename in FILENAMES.items():
    url = URL_BASE + filename
    path = os.path.join(args.data_dir, filename)
    print('Saving %s ...' % path)
    urllib.request.urlretrieve(url, filename=path)


if __name__ == '__main__':
  main()
