import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Lambda, BatchNormalization, Dropout, Layer
from tensorflow.keras.models import Model

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

def BiLSTM_with_Attention(max_time, n_input, lstm_size, attention_size, keep_prob, n_hidden, n_class):

    # Reshaping input eeg signal for BiLSTM
    input_layer = Input(shape=(n_input,))
    reshaped_layer = Lambda(lambda x: tf.reshape(x, (-1, max_time, n_input)))(input_layer)

    # BiLSTM Model with dropout
    bilstm_layer = Bidirectional(
        LSTM(units=lstm_size, dropout=1 - keep_prob, return_sequences=True)
    )

    outputs = bilstm_layer(reshaped_layer)

    attention_output, alphas = AttentionLayer(attention_size, return_alphas=True)(outputs)

    attention_output = Dropout(1 - keep_prob)(attention_output)

    # Using Keras Dense layer with softplus activation.
    fc1 = Dense(
        n_hidden,
        activation='softplus',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(0.01),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(attention_output)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(1 - keep_prob)(fc1)

    # Second fully-connected layer: output layer with softmax activation.
    fc2 = Dense(
        n_class,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(0.01)
    )(fc1)

    reshaped_output_layer = Lambda(lambda x: tf.repeat(x, repeats=64, axis=0))(fc2)

    model = Model(inputs=input_layer, outputs=reshaped_output_layer)
    return model


