import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path).values
    num_attributes = data.shape[1] - 1  # Última coluna é a classe
    num_classes = len(np.unique(data[:, -1]))
    return data, num_attributes, num_classes

def calculate_hidden_neurons(input_neurons, output_neurons):
    return int(np.sqrt(input_neurons * output_neurons))