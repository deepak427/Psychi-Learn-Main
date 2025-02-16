import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Lambda, BatchNormalization, Dropout, Layer
from tensorflow.keras.models import Model

from Network.Custom_Layers.ReshapeLayer import ReshapeLayer
from Network.Custom_Layers.AttentionLayer import AttentionLayer

def BiLSTM_with_Attention(max_time, n_input, lstm_size, attention_size, drop_out, n_hidden, n_class):

    # Reshaping input eeg signal for BiLSTM
    input_layer = Input(shape=(max_time * n_input,))
    reshaped_layer = ReshapeLayer(max_time, n_input)(input_layer)

    # BiLSTM Model with dropout
    bilstm_layer = Bidirectional(
        LSTM(units=lstm_size, dropout=drop_out, return_sequences=True)
    )

    outputs = bilstm_layer(reshaped_layer)

    attention_output, alphas = AttentionLayer(attention_size, return_alphas=True)(outputs)

    attention_output = Dropout(drop_out)(attention_output)

    # Using Keras Dense layer with softplus activation.
    fc1 = Dense(
        n_hidden,
        activation='softplus',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(0.01),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(attention_output)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(drop_out)(fc1)

    # Second fully-connected layer: output layer with softmax activation.
    fc2 = Dense(
        n_class,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(0.01)
    )(fc1)

    model = Model(inputs=input_layer, outputs=fc2)
    return model


