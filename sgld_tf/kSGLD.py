from __future__ import absolute_import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf



class kSGLDOpt(optimizer.Optimizer):
    """Implementation of SGLD.
    """
    def __init__(self, learning_rate=0.001,decay=0.9, epsilon=1e-10,
                 use_locking=False, name="kSGLDOpt"):
        super(kSGLDOpt, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._decay = decay
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._decay_t = None
        self._epsilon_t = None

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads_and_vars = tuple(grads_and_vars)
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                g = ops.convert_to_tensor_or_indexed_slices(g)
            p = optimizer._get_processor(v)
            converted_grads_and_vars.append((g, v, p))
        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list  = [v for g, v,_ in converted_grads_and_vars if g is not None]
        with ops.control_dependencies(None):
            self._create_slots([optimizer._get_variable_for(v) for v in var_list])
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            for grad, var, processor in converted_grads_and_vars:
                if grad is None:
                    continue
                scope_name = var.op.name
                with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, grad))
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        apply_updates = state_ops.assign_add(global_step, 1, name=name)
            if isinstance(apply_updates, ops.Tensor):
                apply_updates = apply_updates.op
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)
        return apply_updates

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
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        decay_t = math_ops.cast(self._decay_t, var.dtype.base_dtype)
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

