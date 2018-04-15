import tensorflow as tf
import sgld_tf

class CNN:
    def __init__(self,
                 train_data,
                 train_labels,
                 test_data,
                 test_labels,
                 model_dir="/tmp/tf/",
                 optimize_method="sgd",
                 learning_rate=0.001,
                 decay=.96,
                 decay_interval=50,
                 shuffle=True,
                 save_checkpoint_steps=None,
                 keep_checkpoint_max=100,
                 checkpoint_init = None,
                 bs=1024):
        self.lrdecay=decay
        self.decay_interval = decay_interval
        self.train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=bs,
            num_epochs=None,
            shuffle=shuffle)
        self.eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=test_labels,
            batch_size=bs,
            num_epochs=None,
            shuffle=False)
        self.optimize_method = optimize_method
        self.learning_rate = learning_rate
        self.checkpoint_init = checkpoint_init
        self.bs = bs

        if save_checkpoint_steps is not None:
            self.estimator = tf.estimator.Estimator(
                model_fn=self.model_fn,
                model_dir=model_dir,
                config=tf.estimator.RunConfig(
                    save_checkpoints_steps=save_checkpoint_steps,
                    keep_checkpoint_max=keep_checkpoint_max))
        else:
            self.estimator = tf.estimator.Estimator(
                model_fn=self.model_fn,
                model_dir=model_dir)



    def make_conv_layer(self, n_filters, kernel_size, inputs, padding="same"):
        conv = tf.layers.Conv2D(filters=n_filters,
                                kernel_size=kernel_size,
                                padding=padding)
        pre = conv(inputs)
        out = tf.nn.relu(pre)

        return conv, pre, out

    def make_dense_layer(self, n_units, inputs):
        dense = tf.layers.Dense(units=n_units)
        pre = dense(inputs)
        out = tf.nn.relu(pre)

        return dense, pre, out

    def model_fn(self, features, labels, mode):
        """Model function for CNN."""
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        conv1, conv1_pre, conv1_out = self.make_conv_layer(32, [5, 5], input_layer)
        pool1 = tf.layers.max_pooling2d(inputs=conv1_out, pool_size=[2, 2], strides=2)

        conv2, conv2_pre, conv2_out = self.make_conv_layer(64, [5, 5], pool1)
        pool2 = tf.layers.max_pooling2d(inputs=conv2_out, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense, dense_pre, dense_out = self.make_dense_layer(1024, pool2_flat)
        dense2, logits, dense2_out = self.make_dense_layer(10, dense_out)

        predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if self.checkpoint_init:
            collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assignment = {v.name.split(':')[0]: v for v in collection}
            tf.train.init_from_checkpoint(self.checkpoint_init, assignment)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # register layers with kfac:
        if self.optimize_method == "kfac" or self.optimize_method=="ksgld":
            layer_collection = tf.contrib.kfac.layer_collection.LayerCollection()
            layer_collection.register_conv2d((conv1.kernel, conv1.bias), (1, 1, 1, 1), "SAME", input_layer, conv1_pre)
            layer_collection.register_conv2d((conv2.kernel, conv2.bias), (1, 1, 1, 1), "SAME", pool1, conv2_pre)
            layer_collection.register_fully_connected((dense.kernel, dense.bias), pool2_flat, dense_pre)
            layer_collection.register_fully_connected((dense2.kernel, dense2.bias), dense_out, logits)
            layer_collection.register_categorical_predictive_distribution(logits, name="logits")



        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.optimize_method == "kfac":
                optimizer = tf.contrib.kfac.optimizer.KfacOptimizer(
                    learning_rate=self.learning_rate,
                    damping=0.0001,
                    cov_ema_decay=0.95,
                    layer_collection=layer_collection)
            elif self.optimize_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
            elif self.optimize_method == "sgld":
                optimizer = sgld_tf.SGLD(
                    lrdecay=self.lrdecay,
                    decay_interval=self.decay_interval,
                    learning_rate=self.learning_rate)
            elif self.optimize_method == "psgld":
                optimizer = sgld_tf.pSGLD(
                    learning_rate=self.learning_rate,
                    lrdecay=self.lrdecay,
                    decay_interval=self.decay_interval
                )
            elif self.optimize_method == "ksgld":
                optimizer = sgld_tf.kSGLDOpt(
                    lrdecay=self.lrdecay,
                    decay_interval=self.decay_interval,
                    learning_rate=self.learning_rate,
                decay=0.9, layer_collection = layer_collection)

            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {"accuracy": tf.metrics.accuracy(
                  labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

