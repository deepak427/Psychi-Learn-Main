import numpy as np
import pandas as pd


def DatasetLoader(DIR):

    # Read Training Data and Labels
    train_data = pd.read_csv(DIR + '\\training_set.csv', header=None)
    train_data = np.array(train_data).astype('float32')

    train_labels = pd.read_csv(DIR + '\\training_label.csv', header=None)
    train_labels = np.array(train_labels).astype('float32')

    # If you met the below error:
    # ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'
    # Then you have to uncomment the below code:
    # train_labels = np.squeeze(train_labels)
    
    # Read Testing Data and Labels
    test_data = pd.read_csv(DIR + '\\test_set.csv', header=None)
    test_data = np.array(test_data).astype('float32')

    test_labels = pd.read_csv(DIR + '\\test_label.csv', header=None)
    test_labels = np.array(test_labels).astype('float32')
    # If you met the below error:
    # ValueError: Cannot feed value of shape (1024, 1) for Tensor 'input/label:0', which has shape '(1024,)'
    # Then you have to uncomment the below code:
    # test_labels = np.squeeze(test_labels)

    return train_data, train_labels, test_data, test_labels