import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class AttentionLayer(Layer):

    def __init__(self, attention_size, return_alphas=False, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_size = attention_size
        self.return_alphas = return_alphas

    def build(self, input_shape):
        # input_shape should be: (batch_size, time_steps, features)
        self.w_omega = Dense(self.attention_size, activation="tanh", name="att_dense_tanh")
        self.u_omega = Dense(1, name="att_dense_linear")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        v = self.w_omega(inputs)  # Shape: (batch_size, time_steps, attention_size)
        vu = self.u_omega(v)      # Shape: (batch_size, time_steps, 1)
        
        # Compute the attention weights by applying softmax on the time dimension
        alphas = tf.nn.softmax(vu, axis=1)  # Shape: (batch_size, time_steps, 1)
        
        output = tf.reduce_sum(inputs * alphas, axis=1)  # Shape: (batch_size, features)
        
        if self.return_alphas:
            return output, alphas
        return output
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[-1])
        if self.return_alphas:
            alphas_shape = (input_shape[0], input_shape[1], 1)
            return [output_shape, alphas_shape]
        return output_shape

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            "attention_size": self.attention_size,
            "return_alphas": self.return_alphas,
        })
        return config