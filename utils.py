import numpy as np


def tokens_to_ids(tokens, token_vocab):
  """ Convert tokens to integer IDs.

  `token_vocab` is a list for convenience. This isn't too inefficient because
  we build a dictionary for any specific `token_vocab` and cache it for future
  calls.

  Args:
    tokens: A list of tokens.
    token_vocab: A list of unique tokens serving as the vocabulary.

  Returns:
    A list of integers with the same length as `tokens`. The integers range
    from `0` to `len(token_vocab) - 1`.
  """
  token_vocab = tuple(token_vocab)
  if not hasattr(tokens_to_ids, 'token_to_id_dicts'):
    tokens_to_ids.token_to_id_dicts = {}
  if token_vocab not in tokens_to_ids.token_to_id_dicts:
    tokens_to_ids.token_to_id_dicts[token_vocab] = {token: i for (i, token) in enumerate(token_vocab)}
  token_to_id_dict = tokens_to_ids.token_to_id_dicts[token_vocab]
  ids = [token_to_id_dict[token] for token in tokens]
  return ids


def ids_to_tokens(ids, token_vocab):
  """ Convert integer IDs to tokens.

  Args:
    ids: A list of integers.
    token_vocab: A list of unique tokens serving as the vocabulary.

  Returns:
    A list of tokens.
  """
  tokens = [token_vocab[i] for i in ids]
  return tokens


def one_hot(labels, num_classes):
  """ Get one-hot encodings.

  Args:
    labels: An int or float NumPy array with values from 0 to
      `num_classes - 1`. Any shape is allowed. `labels` can also contain NaNs;
      in this case, the corresponding encoding will be all NaNs.
    num_classes: An integer.

  Returns:
    A float NumPy array with shape `labels.shape + [num_classes]`.
  """
  labels = np.asarray(labels)
  flat_labels = labels.flatten()
  nan_mask = np.isnan(flat_labels)
  nan_ind, = np.nonzero(nan_mask)
  non_nan_ind, = np.nonzero(~nan_mask)
  encodings_matrix = np.zeros([flat_labels.size, num_classes], dtype=np.float)
  encodings_matrix[nan_ind, :] = np.nan
  encodings_matrix[non_nan_ind, flat_labels[~nan_mask].astype(np.int)] = 1.0
  encodings = np.reshape(encodings_matrix, list(labels.shape) + [num_classes])
  return encodings


def seq_ind_generator(num_seqs, shuffle=True):
  """ A sequence-index generator.

  Args:
    num_seqs: An integer. The number of sequences we'll be indexing.
    shuffle: A boolean. If true, randomly shuffle indices epoch by epoch.

  Yields:
    An integer in `[0, num_seqs)`.
  """

  seq_inds = list(range(num_seqs))
  while True:
    if shuffle:
      np.random.shuffle(seq_inds)
    for seq_ind in seq_inds:
      yield seq_ind


def pad_seq(seq, length, value):
  """ Pad a sequence to have a specified length.

  Args:
    seq: A 2-D NumPy array with shape `[seq_length, size]`.
    length: The length of the sequence once padded.
    value: The value to pad extra time steps with. `seq` is required to have
      the same data type as `value`.

  Returns:
    A 2-D NumPy array with shape `[length, size]`.
  """
  seq = np.asarray(seq)
  if seq.dtype != type(value):
    raise ValueError('seq and value have different data types %s and %s.' % (seq.dtype, type(value)))
  padded_seq = np.pad(seq, [[0, length - len(seq)], [0, 0]], mode='constant', constant_values=value)
  return padded_seq


def full_bptt_batch_generator(input_seqs, target_seqs, max_batch_size, num_epochs=np.inf,
                              shuffle=True, random_roll=False):
  """ Generate batches for full backpropagation through time.

  Each input sequence must have the same length as its corresponding target
  sequence. (If there are fewer targets, insert nans to align the existing
  targets with their corresponding time steps.)

  All sequences in a batch will be cast to floats. To make sequence lengths
  the same for an individual batch, inputs are padded with 0.0 and targets
  are padded with nan.

  TODO: Make this more efficient. This isn't important for now because the time
  spent here is typically negligible compared to the time spent in a single
  forward, backward pass.

  Args:
    input_seqs: A list of 2-D NumPy arrays, each with shape
      `[seq_length, input_size]`.
    target_seqs: A list of 2-D NumPy arrays, each with shape
      `[seq_length, target_size]`.
    max_batch_size: An integer. Smaller batch sizes are possible when
      `max_batch_size` doesn't divide the total number of sequences that we'll
      end up visiting in `num_epochs` epochs.
    num_epochs: An integer or float. The number of epochs to visit.
    shuffle: A boolean. If True, shuffle sequences epoch by epoch.
    random_roll: A boolean. If True, randomly roll each input, target
      sequence along the time axis.

  Yields:
    A tuple `(batch_inputs, batch_targets)`, each a 3-D float NumPy array with
    shape `[batch_size, batch_length, size]`.
  """
  num_seqs = len(input_seqs)
  num_seqs_to_visit = num_seqs * num_epochs
  seq_ind_gen = seq_ind_generator(num_seqs, shuffle=shuffle)
  num_seqs_visited = 0
  while num_seqs_visited < num_seqs_to_visit:
    batch_size = min(max_batch_size, num_seqs_to_visit - num_seqs_visited)
    batch_inds = [next(seq_ind_gen) for _ in range(batch_size)]
    batch_seq_lengths = [len(input_seqs[i]) for i in batch_inds]
    batch_length = np.max(batch_seq_lengths)
    batch_inputs, batch_targets = [], []
    for i in batch_inds:
      input_seq, target_seq = input_seqs[i].astype(np.float), target_seqs[i].astype(np.float)
      if len(input_seq) != len(target_seq):
        raise ValueError('Input sequence length does not match target sequence length.')
      if random_roll:
        shift = np.random.randint(0, len(input_seq))
        input_seq, target_seq = np.roll(input_seq, shift, axis=0), np.roll(target_seq, shift, axis=0)
      input_seq, target_seq = pad_seq(input_seq, batch_length, 0.0), pad_seq(target_seq, batch_length, np.nan)
      batch_inputs.append(input_seq)
      batch_targets.append(target_seq)
    num_seqs_visited += batch_size
    yield np.asarray(batch_inputs), np.asarray(batch_targets)


def truncated_bptt_batch_generator(input_seqs, target_seqs, max_batch_size, max_truncation_length, num_epochs=np.inf,
                                   shuffle=True, random_roll=False):
  """ Generate batches for truncated backpropagation through time.

  All sequences in a batch will be cast to floats. To make sequence lengths
  the same for an individual batch, inputs are padded with 0.0 and targets
  are padded with nan.

  TODO: Make this more efficient. In particular, when sequence lengths vary
  dramatically within a dataset, we should (a) maintain `batch_size` different
  full-BPTT generators, (b) start a new sequence immediately after another is
  exhausted, and (c) maintain sequence-level resets instead of batch-level
  resets. This would also require a model that can handle such sequence-level
  resets.

  Args:
    input_seqs: A list of 2-D NumPy arrays, each with shape
      `[seq_length, input_size]`.
    target_seqs: A list of 2-D NumPy arrays, each with shape
      `[seq_length, target_size]`.
    max_batch_size: An integer. Smaller batch sizes are possible when
      `max_batch_size` doesn't divide the total number of sequences that we'll
      end up visiting in `num_epochs` epochs.
    max_truncation_length: An integer. Shorter lengths are possible when
      `max_truncation_length` doesn't evenly divide all sequence lengths.
    num_epochs: An integer or float. The number of epochs to visit.
    shuffle: A boolean. If True, shuffle sequences epoch by epoch.
    random_roll: A boolean. If True, randomly roll each input, target
      sequence along the time axis (of course before splitting the sequence
      up for truncated backprop).

  Yields:
    A tuple `(batch_inputs, batch_targets, reset_states)`. `batch_inputs` and
    `batch_targets` are both 3-D float NumPy arrays with shape
    `[batch_size, batch_length, size]`. `reset_states` is a boolean
    indicating whether states should be reset before processing the batch.
  """

  full_bptt_batch_gen = full_bptt_batch_generator(input_seqs, target_seqs, max_batch_size, num_epochs,
                                                  shuffle=shuffle, random_roll=random_roll)
  for batch_inputs, batch_targets in full_bptt_batch_gen:
    length = batch_inputs.shape[1]
    split_ind = list(range(max_truncation_length, length, max_truncation_length))
    split_batch_inputs = np.array_split(batch_inputs, split_ind, axis=1)
    split_batch_targets = np.array_split(batch_targets, split_ind, axis=1)
    for i, (batch_inputs, batch_targets) in enumerate(zip(split_batch_inputs, split_batch_targets)):
      yield batch_inputs, batch_targets, i == 0
