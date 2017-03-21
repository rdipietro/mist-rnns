import numpy as np
import tensorflow as tf
import layers


class RNNModel(object):
  """ Base class for recurrent neural network models.

  All weight matrices are initialized to have normally-distributed elements
  from N(0, 1 / num_hidden_units), and by default all biases are initialized
  to 0.

  Args:
    layer_type: A string. See `layers`.
    input_size: An integer.
    target_size: An integer.
    num_hidden_units: An integer.
    activation_type: A string. See `tensorflow.nn`.
    **kwargs: Keyword arguments that are passed to the RNN layer. See `layers`.

  Attributes:
    inputs: A 3-D float32 Placeholder with shape
      `[batch_size, length, input_size]`. Batch size and length are allowed to
      vary. Invalid time steps should be padded with 0.0 (which will then
      simply yield invalid outputs for these time steps).
    targets: A 3-D float32 Placeholder with shape
      `[batch_size, length, target_size]`. The batch size and target
      length should be the same as those of `inputs`. NaN values denote time
      steps without targets.
    initial_rnn_states: A 3-D float32 Tensor with shape
      `[batch_size, num_initial_states, state_size]`. This defaults to all
      zeros but can be replaced by feeding other values.
    final_rnn_states: A 3-D float32 Tensor with the same shape as
      `initial_rnn_states`.
    valid_split_ind: A 1-D int32 Tensor with shape `[batch_size - 1]`. These
      are the split indices (e.g. to be used with `np.array_split`) that
      separate `valid_targets` and `valid_predictions` into a list of
      sequences, each with an unequal number of time steps.
    valid_targets: A 2-D float32 Tensor with shape
      `[num_valid_steps, target_size]`.
    valid_predictions: A 2-D float32 Tensor with the shape
      `[num_valid_steps, target_size]`.
    valid_stepwise_loss: A 0-D float32 Tensor. This treats each time step as an
      example and forms the final loss by averaging across all valid time steps
      in the batch.
    valid_stepwise_loss_for_opt: A 0-D float32 Tensor. Valid stepwise loss,
      scaled by the number of valid sequences in the batch.
    valid_seq_losses: A 1-D float32 Tensor with shape `[num_valid_seqs]`.
    valid_seqwise_loss: A 0-D float32 Tensor. This treats each sequence as an
      example and forms the final loss by averaging across all valid sequences
      in the batch. Note: This often yields good results either way, but we
      need to scale this loss if we'd like batches with fewer sequences to
      result in correctly-scaled gradients.
    valid_seqwise_loss_for_opt: A 0-D float32 Tensor. Valid seqwise loss,
      scaled by the valid number of sequences in the batch.
  """

  def __init__(self, layer_type, input_size, target_size, num_hidden_units, activation_type,
               **kwargs):

    self.input_size = input_size
    self.target_size = target_size
    self.num_hidden_units = num_hidden_units
    self.square_initializer = tf.random_normal_initializer(0.0, np.sqrt(1.0 / num_hidden_units))
    self.non_square_initializer = tf.random_normal_initializer(0.0, np.sqrt(1.0 / num_hidden_units))
    self.bias_initializer = tf.constant_initializer(0.0)
    Layer = getattr(layers, layer_type)
    activation = getattr(tf.nn, activation_type)

    self.inputs = tf.placeholder(tf.float32, shape=[None, None, input_size], name='inputs')
    self.targets = tf.placeholder(tf.float32, shape=[None, None, target_size], name='targets')
    self.batch_size = tf.shape(self.inputs)[0]
    self.length = tf.shape(self.inputs)[1]

    valid_mask_incl_invalid_seqs = tf.logical_not(tf.is_nan(self.targets[0:, 0:, 0]))
    target_step_counts = tf.reduce_sum(tf.to_int32(valid_mask_incl_invalid_seqs), axis=[1],
                                       name='target_step_counts')
    valid_seq_mask = tf.greater(target_step_counts, 0, name='valid_seq_mask')
    self.valid_split_ind = tf.identity(tf.cumsum(target_step_counts)[:-1], name='valid_split_ind')
    valid_seq_ids_incl_invalid_seqs = tf.tile(tf.expand_dims(tf.range(0, self.batch_size), 1), [1, self.length])
    valid_seq_ids = tf.boolean_mask(valid_seq_ids_incl_invalid_seqs, valid_mask_incl_invalid_seqs,
                                         name='valid_seq_ids')
    self.valid_targets = tf.boolean_mask(self.targets, valid_mask_incl_invalid_seqs, name='valid_targets')

    with tf.variable_scope('rnn') as rnn_scope:
      inputs = self.inputs
      self._rnn_layer = Layer(inputs, self.num_hidden_units, activation, self.square_initializer,
                              self.non_square_initializer, self.bias_initializer, **kwargs)
      self.initial_rnn_states = self._rnn_layer.initial_states
      self.final_rnn_states = self._rnn_layer.final_states

    with tf.variable_scope('predictions') as predictions_scope:
      W = tf.get_variable('W', shape=[self.num_hidden_units, self.target_size], initializer=self.non_square_initializer)
      b = tf.get_variable('b', shape=[self.target_size], initializer=self.bias_initializer)
      valid_rnn_outputs = tf.boolean_mask(self._rnn_layer.outputs, valid_mask_incl_invalid_seqs)
      self.valid_predictions = tf.nn.xw_plus_b(valid_rnn_outputs, W, b, name = 'valid_predictions')

    with tf.variable_scope('loss'):

      num_valid_seqs = tf.reduce_sum(tf.to_float(valid_seq_mask))

      stepwise_losses = self._compute_stepwise_losses()
      self.valid_stepwise_loss = tf.reduce_mean(stepwise_losses, name='stepwise_loss')
      self.valid_stepwise_loss_for_opt = tf.identity(num_valid_seqs * self.valid_stepwise_loss,
                                                     name='valid_stepwise_loss_for_opt')

      time_counts = tf.to_float(tf.expand_dims(target_step_counts, 1)) * tf.to_float(valid_mask_incl_invalid_seqs)
      valid_time_counts = tf.boolean_mask(time_counts, valid_mask_incl_invalid_seqs)
      seq_losses = tf.unsorted_segment_sum(stepwise_losses / valid_time_counts, valid_seq_ids, self.batch_size)
      self.valid_seq_losses = tf.boolean_mask(seq_losses, valid_seq_mask, name='valid_seq_losses')
      self.valid_seqwise_loss = tf.reduce_mean(self.valid_seq_losses, name='valid_seqwise_loss')
      self.valid_seqwise_loss_for_opt = tf.identity(num_valid_seqs * self.valid_seqwise_loss,
                                                    name='valid_seqwise_loss_for_opt')

  def _compute_stepwise_losses(self):
    """ Returns a 1-D float32 Tensor with shape `[num_valid_steps]`. """
    raise NotImplementedError()


class RNNClassificationModel(RNNModel):
  """ An RNN classification model.

  Each target time step consists of a single multinomial distribution.
  """
  def _compute_stepwise_losses(self):
    losses = tf.losses.softmax_cross_entropy(onehot_labels=self.valid_targets, logits=self.valid_predictions)
    return losses


class RNNRegressionModel(RNNModel):
  """ An RNN regression model.

  Each target time step consists of N real numbers.
  """
  def _compute_stepwise_losses(self):
    losses = tf.pow(self.valid_predictions - self.valid_targets, 2)
    losses = tf.reduce_sum(losses, axis=[1])
    return losses


class RNNMultiLabelClassificationModel(RNNModel):
  """ An RNN multi-label classification model.

  Each target time step consists of N binomial distributions.
  """
  def _compute_stepwise_losses(self):
    losses = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.valid_targets, logits=self.valid_predictions)
    losses = tf.reduce_sum(losses, axis=[1])
    return losses
