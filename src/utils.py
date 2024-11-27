import pandas as pd
import numpy as np

def load_data(training_file):
    data = pd.read_csv(training_file)
    X = data.iloc[:, :-1].values  # Input features (all except last column)
    y = data.iloc[:, -1].values  # Target labels (last column)
    return X, y

def get_neuron_counts(data):
    input_neurons = data['train'].shape[1] - 1  # Excluindo a coluna de classe
    output_neurons = data['train']['classe'].nunique()
    return input_neurons, output_neurons

def calculate_hidden_neurons(input_neurons, output_neurons):
    return int(np.sqrt(input_neurons * output_neurons))

def encode_labels(y, num_classes, activation_function):
    """
    Encode labels for the neural network.
    - Logistic: One-hot encoding
    - Tanh: Bipolar encoding
    """
    encoded = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        if activation_function == 'logística':
            encoded[i, label - 1] = 1  # One-hot encoding
        elif activation_function == 'tangente':
            encoded[i] = -1  # All -1 initially
            encoded[i, label - 1] = 1  # Set the class position to +1
    return encoded
