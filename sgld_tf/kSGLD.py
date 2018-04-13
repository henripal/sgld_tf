from __future__ import absolute_import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.contrib.kfac.python.ops import estimator as est
from tensorflow.python.ops import variables as tf_variables
import tensorflow as tf


class kSGLDOpt(optimizer.Optimizer):
    """Implementation of SGLD.
    """
    def __init__(self, learning_rate=0.001,decay=0.9, epsilon=1e-10,
                 damping = 0.001, cov_ema_decay=0.95,
                 layer_collection = None, estimation_mode = 'gradients',
                 colocate_gradient_with_ops = True,
                 use_locking=False, name="kSGLDOpt"):
        super(kSGLDOpt, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._decay = decay
        self._epsilon = epsilon

        self._variables = tf_variables.trainable_variables()
        self.damping_fn = lambda: damping
        self.cov_ema_decay = cov_ema_decay
        self.layer_collection = layer_collection
        self.estimation_mode = estimation_mode
        self.colocate_gradient_with_ops = colocate_gradient_with_ops

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._decay_t = None
        self._epsilon_t = None

        self._fisher_est = est.FisherEstimator(self.damping_fn,
                                               self._variables,
                                               self.cov_ema_decay,
                                               self.layer_collection,
                                               self.estimation_mode,
                                               self.colocate_gradient_with_ops)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads_and_vars = tuple(grads_and_vars)
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                g = ops.convert_to_tensor_or_indexed_slices(g)
            # p = optimizer._get_processor(v)
            converted_grads_and_vars.append((g, v))


        converted_grads_and_vars = list(converted_grads_and_vars)
        var_list  = [v for g, v in converted_grads_and_vars if g is not None]
        with ops.control_dependencies(None):
            self._create_slots([optimizer._get_variable_for(v) for v in var_list])

        with ops.name_scope(name, self._name) as name:
            self._prepare()
            update_ops = self._apply_dense(converted_grads_and_vars)
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
        # Create slot for the Langevin Noise
        for v in var_list:
            init_noise = init_ops.random_normal_initializer(dtype=v.dtype.base_dtype)
            self._get_or_make_slot_with_initializer(v, init_noise, v.get_shape(),
                                                v.dtype.base_dtype, "noise",
                                                self._name)

    def _apply_dense(self, grads_and_vars):
        update_ops = []
        noise_and_vars = []
        for grad, var in grads_and_vars:
            noise = self.get_slot(var, "noise" )
            noise_t = noise.assign(tf.random_normal(shape=tf.shape(noise)))
            noise_and_vars.append((noise_t, var))

        preconditioned_noise_and_vars = self._fisher_est.multiply_inverse(noise_and_vars)
        # preconditioned_noise_and_vars = noise_and_vars

        for (grad, var), (pnoise, _) in zip(grads_and_vars, preconditioned_noise_and_vars):
            if grad is None:
                continue
            scope_name = var.op.name
            with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
                var_update = state_ops.assign_sub(var, lr_t*grad - lr_t * lr_t * pnoise)
                update_ops.append(control_flow_ops.group(*[var_update, pnoise]))
        # decay_t = math_ops.cast(self._decay_t, var.dtype.base_dtype)
        # epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)



        # var_update = state_ops.assign_sub(
        #     var,lr_t*grad + lr_t*lr_t*tf.divide(tf.random_normal(shape=tf.shape(grad)), m))
        # var_update = state_ops.assign_sub(
        #      var,lr_t*grad)
        #Update 'ref' by subtracting 'value
        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return update_ops

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

