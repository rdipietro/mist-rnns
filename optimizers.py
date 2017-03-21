import tensorflow as tf


def clip(grads_and_vars, max_global_norm):
  """ Clip the gradients that are returned from a TensorFlow Optimizer.

  Note that the term "clipping" is often used in literature but here is actually
  the wrong term: if the norm of all gradients concatenated does not exceed
  `max_global_norm`, then don't modify them. If the norm does exceed
  `max_global_norm`, then rescale all gradients globally so that the new norm
  becomes `max_global_norm`.

  Args:
    grads_and_vars: A list of `(grad, var)` pairs.
    max_global_norm: A float.

  Returns:
    A list of `(grad, var)` pairs with clipped gradients.
  """

  grads, vars = zip(*grads_and_vars)
  grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
  grads_and_vars = list(zip(grads, vars))
  return grads_and_vars


class ClippingGradientDescentOptimizer(tf.train.GradientDescentOptimizer):
  """ Construct a GradientDescentOptimizer with gradient clipping. """

  def __init__(self, learning_rate, max_global_norm=1.0, **kwargs):
    super().__init__(learning_rate, **kwargs)
    self.max_global_norm = max_global_norm

  def compute_gradients(self, loss, **kwargs):
    grads_and_vars = super().compute_gradients(loss, **kwargs)
    return clip(grads_and_vars, self.max_global_norm)


class ClippingMomentumOptimizer(tf.train.MomentumOptimizer):
  """ Construct a MomentumOptimizer with gradient clipping. """

  def __init__(self, learning_rate, momentum=0.9, max_global_norm=1.0, **kwargs):
    super().__init__(learning_rate, momentum, **kwargs)
    self.max_global_norm = max_global_norm

  def compute_gradients(self, loss, **kwargs):
    grads_and_vars = super().compute_gradients(loss, **kwargs)
    return clip(grads_and_vars, self.max_global_norm)
