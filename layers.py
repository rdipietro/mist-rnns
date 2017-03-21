import numpy as np
import tensorflow as tf
import utils


class RNNLayer(object):
  """ Base class for recurrent neural network layers.

  Note that inputs may be padded with 0.0s to allow batch processing of
  sequences with unequal lengths. In this case the corresponding outputs and
  states (including final states) will be invalid after a certain number of
  time steps. It's up to the user to handle these invalid entries
  appropriately, for example by resetting states in the case of truncated
  backpropagation through time.

  Args:
    inputs: A 3-D float32 Tensor with shape `[batch_size, length, input_size]`.
    num_hidden_units: An integer.
    activation: An element-wise TensorFlow function.
    square_initializer: An initializer for square weight matrices.
    non_square_initializer: An initializer for non square weight matrices.
    bias_initializer: An initializer to be used for bias terms.
    **kwargs: These simply get appended as attributes, to be handled as needed
      by inheriting classes.

  Attributes:
    num_initial_states: An integer.
    initial_states: A 3-D float32 Tensor with shape
      `[batch_size, num_initial_states, state_size]`. This defaults to all
      zeros but can be replaced by feeding other other values.
    outputs: A 3-D float32 Tensor with shape
      `[batch_size, length, num_hidden_units]`.
    states: A 3-D float32 Tensor with shape
      `[batch_size, length, state_size]`.
    final_states: A 3-D float32 Tensor with the same shape as
      `initial_states`.
  """

  def __init__(self, inputs, num_hidden_units, activation, square_initializer, non_square_initializer,
               bias_initializer, **kwargs):

    self.inputs = inputs
    self.num_hidden_units = num_hidden_units
    self.activation = activation
    self.square_initializer = square_initializer
    self.non_square_initializer = non_square_initializer
    self.bias_initializer = bias_initializer
    for key, value in kwargs.items():
      setattr(self, key, value)

    # Batch size and length are dynamic. Input size isn't.
    self.batch_size = tf.shape(self.inputs)[0]
    self.length = tf.shape(self.inputs)[1]
    self.input_size = inputs.get_shape().as_list()[2]
    self.initial_states = self._get_default_initial_states()
    self.num_initial_states = self.initial_states.get_shape().as_list()[1]
    self.outputs, self.states = self._compute_states()
    self.final_states = self.states[:, -self.num_initial_states:, :]

  def _get_default_initial_states(self):
    """ Return default initial states.

    Returns:
      A 3-D float32 Tensor with shape
      `[batch_size, num_initial_states, state_size]`.
    """
    return tf.zeros([self.batch_size, 1, self.num_hidden_units])

  def _linear(self, h, x, output_size, shift=0.0, scope=None):
    """ output = h W_h + x W_x + b + shift

    Args:
      h: A 2-D float32 Tensor with shape `[batch_size, num_hidden_units]`.
      x: A 2-D float32 Tensor with shape `[batch_size, input_size]`.
      output_size: An integer.
      shift: A float.
      scope: A TensorFlow scope or string.

    Returns:
      A 2-D float32 Tensor with shape `[batch_size, output_size]`.
    """
    num_units, input_size = self.num_hidden_units, self.input_size
    scope = scope or tf.get_variable_scope()
    W_h_init = self.square_initializer if output_size == num_units else self.non_square_initializer
    W_x_init = self.square_initializer if output_size == input_size else self.non_square_initializer
    with tf.variable_scope(scope):
      b = tf.get_variable('b', shape=[output_size], initializer=self.bias_initializer) + shift
      W_h = tf.get_variable('W_h', shape=[num_units, output_size], initializer=W_h_init)
      W_x = tf.get_variable('W_x', shape=[input_size, output_size], initializer=W_x_init)
      output = tf.matmul(h, W_h) + tf.nn.xw_plus_b(x, W_x, b)
    return output

  def _compute_states(self):
    """ Compute hidden states.

    Returns:
      A tuple, (outputs, states).
    """
    raise NotImplementedError()


class SimpleLayer(RNNLayer):
  """ A simple RNN layer.

  See `RNNLayer`.
  """

  def _compute_states(self):
    """ Compute hidden states.

    Returns:
      A tuple, (outputs, states).
    """

    _inputs = tf.transpose(self.inputs, [1, 0, 2])
    x_ta = tf.TensorArray(tf.float32, size=self.length).unstack(_inputs)
    h_ta = tf.TensorArray(tf.float32, size=self.length)

    def cond(t, h, h_ta):
      return tf.less(t, self.length)

    def body(t, h, h_ta):

      x = x_ta.read(t)
      num_units, input_size = self.num_hidden_units, self.input_size

      with tf.variable_scope('simple_rnn'):
        h_new = self.activation(self._linear(h, x, num_units, scope='simple_rnn'))

      h_ta_new = h_ta.write(t, h_new)
      return t + 1, h_new, h_ta_new

    t = tf.constant(0)
    h = tf.squeeze(self.initial_states, [1])
    _, _, h_ta = tf.while_loop(cond, body, [t, h, h_ta])

    states = tf.transpose(h_ta.stack(), [1, 0, 2], name='states')
    outputs = tf.identity(states, name='outputs')
    return outputs, states


class LSTMLayer(RNNLayer):
  """ An LSTM layer.

  See `RNNLayer`.

  **kwargs:
    optional_bias_shift: A float. For LSTM, this is often referred to as the
      forget-gate bias. 1.0 by default.
  """

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('optional_bias_shift', 1.0)
    super().__init__(*args, **kwargs)

  def _get_default_initial_states(self):
    return tf.zeros([self.batch_size, 1, 2 * self.num_hidden_units])

  def _compute_states(self):

    _inputs = tf.transpose(self.inputs, [1, 0, 2])
    x_ta = tf.TensorArray(tf.float32, size=self.length).unstack(_inputs)
    h_ta = tf.TensorArray(tf.float32, size=self.length)
    c_ta = tf.TensorArray(tf.float32, size=self.length)

    def cond(t, c, h, c_ta, h_ta):
      return tf.less(t, self.length)

    def body(t, c, h, c_ta, h_ta):

      x = x_ta.read(t)
      num_units, input_size = self.num_hidden_units, self.input_size

      with tf.variable_scope('lstm'):
        c_tilde = self.activation(self._linear(h, x, num_units, scope='c'))
        i = tf.nn.sigmoid(self._linear(h, x, num_units, scope='i'))
        f = tf.nn.sigmoid(self._linear(h, x, num_units, shift=self.optional_bias_shift, scope='f'))
        o = tf.nn.sigmoid(self._linear(h, x, num_units, scope='o'))
        c_new = i * c_tilde + f * c
        h_new = o * self.activation(c_new)

      c_ta_new = c_ta.write(t, c_new)
      h_ta_new = h_ta.write(t, h_new)
      return t + 1, c_new, h_new, c_ta_new, h_ta_new

    t = tf.constant(0)
    c, h = tf.split(tf.squeeze(self.initial_states, [1]), 2, axis=1)
    _, _, _, c_ta, h_ta = tf.while_loop(cond, body, [t, c, h, c_ta, h_ta])

    outputs = tf.transpose(h_ta.stack(), [1, 0, 2], name='outputs')
    cells = tf.transpose(c_ta.stack(), [1, 0, 2])
    states = tf.concat([cells, outputs], axis=2, name='states')
    return outputs, states


class GRULayer(RNNLayer):
  """ A GRU layer.

  See `RNNLayer`.

  **kwargs:
    optional_bias_shift: A float. The initial bias shift for the update gate.
      1.0 by default.
  """

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('optional_bias_shift', 1.0)
    super().__init__(*args, **kwargs)

  def _compute_states(self):

    _inputs = tf.transpose(self.inputs, [1, 0, 2])
    x_ta = tf.TensorArray(tf.float32, size=self.length).unstack(_inputs)
    h_ta = tf.TensorArray(tf.float32, size=self.length)

    def cond(t, h, h_ta):
      return tf.less(t, self.length)

    def body(t, h, h_ta):

      x = x_ta.read(t)
      num_units, input_size = self.num_hidden_units, self.input_size

      with tf.variable_scope('gru'):
        r = tf.nn.sigmoid(self._linear(h, x, num_units, scope='r'))
        h_pre_act = r * h
        h_tilde = self.activation(self._linear(h_pre_act, x, num_units, scope='h'))

        z = tf.nn.sigmoid(self._linear(h, x, num_units, shift=self.optional_bias_shift, scope='z'))
        h_new = z * h + (1 - z) * h_tilde

      h_ta_new = h_ta.write(t, h_new)
      return t + 1, h_new, h_ta_new

    t = tf.constant(0)
    h = tf.squeeze(self.initial_states, [1])
    _, _, h_ta = tf.while_loop(cond, body, [t, h, h_ta])

    states = tf.transpose(h_ta.stack(), [1, 0, 2], name='states')
    outputs = tf.identity(states, name='outputs')
    return outputs, states


class MISTLayer(RNNLayer):
  """ A mixed-history RNN layer.

  See `RNNLayer`.

  **kwargs:
    pre_act_mixture_delays: A list of unique, positive integers.
      `[1, 2, 4, 8, 16, 32, 64, 128]` by default.
  """

  def __init__(self, *args, **kwargs):
    kwargs.setdefault('pre_act_mixture_delays', 2 ** np.arange(0, 8))
    super().__init__(*args, **kwargs)

  def _get_default_initial_states(self):
    return tf.zeros([self.batch_size, np.max(self.pre_act_mixture_delays), self.num_hidden_units])

  def _compute_states(self):

    _inputs = tf.transpose(self.inputs, [1, 0, 2])
    x_ta = tf.TensorArray(tf.float32, size=self.length).unstack(_inputs)

    h_ta_size = self.num_initial_states + self.length
    initial_states = tf.transpose(self.initial_states, [1, 0, 2])

    # infer_shapes=True is buggy and says that shape (?, num_hidden_units) is incompatible with
    # shape (?, num_hidden_units). I've verified that they both have shape
    # (batch_size, num_hidden_units). To avoid this,  we'll set infer_shape=False and
    # skip the consistency check entirely.
    h_ta = tf.TensorArray(tf.float32, size=h_ta_size, clear_after_read=False, infer_shape=False)
    h_ta = h_ta.unstack(initial_states)

    def cond(t, h_ta):
      return tf.less(t, self.length)

    def body(t, h_ta):

      h = h_ta.read(self.num_initial_states + t - 1)
      x = x_ta.read(t)
      num_units, input_size = self.num_hidden_units, self.input_size

      with tf.variable_scope('pre_act'):

        # Shape [batch_size, pre_act_mixture_delays.size, num_units]
        h_history = tf.transpose(h_ta.gather(self.num_initial_states + t - self.pre_act_mixture_delays), [1, 0, 2])

        # Shape [batch_size, pre_act_mixture_delays.size, 1]
        coefs = tf.expand_dims(self._linear(h, x, self.pre_act_mixture_delays.size, scope='coefs'), 2)
        coefs = tf.nn.softmax(coefs, dim=1)

        # Shape [batch_size, num_units]
        h_pre_act = tf.reduce_sum(coefs * h_history, axis=[1])

        r = tf.nn.sigmoid(self._linear(h, x, num_units, scope='r'))
        h_pre_act = r * h_pre_act

      h_tilde = self.activation(self._linear(h_pre_act, x, num_units, scope='mist'))

      h_ta_new = h_ta.write(self.num_initial_states + t, h_tilde)
      return t + 1, h_ta_new

    t = tf.constant(0)
    _, h_ta = tf.while_loop(cond, body, [t, h_ta])

    all_states = h_ta.stack()
    states = tf.transpose(all_states[self.num_initial_states:], [1, 0, 2], name='states')
    outputs = tf.identity(states, name='outputs')
    return outputs, states
