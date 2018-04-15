from __future__ import absolute_import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class pSGLD(optimizer.Optimizer):
    """Implementation of SGLD.
    """
    def __init__(self, learning_rate=0.001,decay=0.96, epsilon=1e-10,
                 lrdecay=.96,
                 decay_interval=50,
                 use_locking=False, name="pSGLD"):
        super(pSGLD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._decay = decay
        self._lrdecay = lrdecay
        self._decay_interval = decay_interval
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._decay_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._decay_t = ops.convert_to_tensor(self._decay, name="decay_t")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            if v.get_shape().is_fully_defined():
                init_rms = init_ops.ones_initializer(dtype=v.dtype.base_dtype)
            else:
                init_rms = array_ops.ones_like(v)
            self._get_or_make_slot_with_initializer(v, init_rms, v.get_shape(),
                                                v.dtype.base_dtype, "rms",
                                                self._name)

    def _apply_dense(self, grad, var):
        lr_t0 = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        decay_t = math_ops.cast(self._decay_t, var.dtype.base_dtype)
        lr_t = tf.train.exponential_decay(lr_t0, tf.train.get_global_step(),
                                          self._decay_interval, self._lrdecay)
        print(lr_t)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)


        m = self.get_slot(var, "rms")
        m_t = m.assign(
            tf.sqrt(epsilon_t + decay_t * m + (1 - decay_t) * tf.square(grad)))

        var_update = state_ops.assign_sub(
            var,lr_t*grad + lr_t*lr_t*tf.divide(tf.random_normal(shape=tf.shape(grad)), m))
        # var_update = state_ops.assign_sub(
        #      var,lr_t*grad)
        #Update 'ref' by subtracting 'value
        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")



class SGLD(optimizer.Optimizer):
    """Implementation of SGLD.
    """
    def __init__(self, learning_rate=0.001,alpha=0.01,
                 lrdecay=.96, decay_interval=50,
                 beta=0.5, use_locking=False, name="SGLD"):
        super(SGLD, self).__init__(use_locking, name)
        self._lrdecay = lrdecay
        self._decay_interval=decay_interval
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # for v in var_list:
        #     self._zeros_slot(v, "m", self._name)
        pass

    def _apply_dense(self, grad, var):
        lr_t0 = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        lr_t = tf.train.exponential_decay(lr_t0, tf.train.get_global_step(),
                                          self._decay_interval, self._lrdecay)
        # beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        # eps = 1e-7 #cap for moving average

        # m = self.get_slot(var, "m")
        # m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(
            var,lr_t*grad + lr_t*lr_t*tf.random_normal(shape=tf.shape(grad)))
        # var_update = state_ops.assign_sub(
        #      var,lr_t*grad)
        #Update 'ref' by subtracting 'value
        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


