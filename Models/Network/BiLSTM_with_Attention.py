import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense

def attention(inputs, attention_size, return_alphas=False):

    # Dense layers for computing attention scores
    w_omega = Dense(attention_size, activation="tanh")
    u_omega = Dense(1) 

    v = w_omega(inputs)  # Shape: (batch_size, time_steps, attention_size)
    vu = u_omega(v)  # Shape: (batch_size, time_steps, 1)

    alphas = tf.nn.softmax(vu, axis=1)

    output = tf.reduce_sum(inputs * alphas, axis=1)  # Shape: (batch_size, features)

    if return_alphas:
        return output, alphas
    return output


def BiLSTM_with_Attention(max_time, n_input, lstm_size, attention_size, keep_prob, weights_1, biases_1, weights_2, biases_2):

    # Reshaping input eeg signal for BiLSTM
    input_layer = Input(shape=(max_time, n_input))

    # BiLSTM Model with dropout
    bilstm_layer = Bidirectional(
        LSTM(units=lstm_size, dropout=1 - keep_prob, return_sequences=True)
    )

    outputs = bilstm_layer(input_layer)
