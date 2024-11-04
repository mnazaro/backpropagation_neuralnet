import numpy as np

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, transfer_function):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.transfer_function = transfer_function
        self.weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
        self.weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)

    def train(self, data, stop_condition, stop_value):
        # Implementar o treinamento da rede neural
        pass

    def test(self, test_data):
        # Implementar o teste da rede neural
        pass

    def transfer(self, x):
        if self.transfer_function == "logística":
            return 1 / (1 + np.exp(-x))
        elif self.transfer_function == "tangente hiperbólica":
            return np.tanh(x)
        else:
            raise ValueError("Função de transferência inválida")