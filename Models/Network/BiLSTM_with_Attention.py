import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional

def BiLSTM_with_Attention(max_time, n_input, lstm_size, attention_size, keep_prob, weights_1, biases_1, weights_2, biases_2):

    # Reshaping input eeg signal for BiLSTM
    input_layer = Input(shape=(max_time, n_input))

    # BiLSTM Model with dropout
    bilstm_layer = Bidirectional(
        LSTM(units=lstm_size, dropout=1 - keep_prob, return_sequences=True)
    )

    outputs = bilstm_layer(input_layer)
