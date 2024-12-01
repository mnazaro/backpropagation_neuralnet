import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function, self.activation_derivative = self._get_activation_function(activation_function)
        
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))


    def _get_activation_function(self, name):
        if name == 'logística':
            return lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x)
        elif name == 'tangente':
            return np.tanh, lambda x: 1 - x**2
        else:
            raise ValueError("Função de ativação desconhecida!")
        
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.activation_function(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = self.activation_function(self.output_input)

        return self.output

    def backward(self, X, y, learning_rate=0.01):
        output_error = y - self.output
        output_delta = output_error * self.activation_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)

        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

    def train(self, X, y, max_epochs, error_threshold, learning_rate=0.01, log_callback=None):
       for epoch in range(max_epochs):
            self.forward(X)

            total_error = np.mean((y - self.output) ** 2)

            if error_threshold is not None and total_error <= error_threshold:
                if log_callback:
                    log_callback(f"Treinamento finalizado após {epoch + 1} épocas. Erro total: {total_error:.4f}")
                break

            self.backward(X, y, learning_rate)
            if log_callback:
                log_callback(f"Época {epoch + 1} - Erro total: {total_error:.4f}")
       else:
            if log_callback:
                log_callback(f"Treinamento finalizado. Máximo de épocas atingido: {max_epochs}. Erro total: {total_error:.4f}")

    def test(self, X):
        output = self.forward(X)
        if self.activation_function == "tangente":
            return np.where(output > 0, 1, -1)
        else:
            return np.argmax(output, axis=1)

