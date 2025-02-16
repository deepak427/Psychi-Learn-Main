import tensorflow as tf
from tensorflow.keras.layers import Layer

class ReshapeLayer(Layer):
    def __init__(self, max_time, n_input, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.max_time = max_time
        self.n_input = n_input

    def call(self, inputs):
        # Reshape from (batch_size, max_time*n_input) to (batch_size, max_time, n_input)
        return tf.reshape(inputs, (-1, self.max_time, self.n_input))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_time, self.n_input)

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({
            "max_time": self.max_time,
            "n_input": self.n_input
        })
        return config
