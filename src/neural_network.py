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
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.activation_function(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = self.activation_function(self.output_input)

        return self.output

    def backward(self, X, y, learning_rate=0.01):
        # Output error
        output_error = y - self.output
        output_delta = output_error * self.activation_derivative(self.output)

        # Hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_output)

        # Update weights
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

    def train(self, X, y, max_epochs=1000, error_threshold=None, learning_rate=0.01, log_callback=None):
       for epoch in range(max_epochs):
            # Forward pass
            self.forward(X)

            # Calculate total error
            total_error = np.mean((y - self.output) ** 2)

            # Check stopping condition
            if error_threshold is not None and total_error <= error_threshold:
                if log_callback:
                    log_callback(f"Treinamento finalizado após {epoch + 1} épocas. Erro total: {total_error:.4f}")
                break

            # Backward pass
            self.backward(X, y, learning_rate)
            if log_callback:
                log_callback(f"Época {epoch + 1} - Erro total: {total_error:.4f}")
       else:
            if log_callback:
                log_callback(f"Treinamento finalizado. Máximo de épocas atingido: {max_epochs}. Erro total: {total_error:.4f}")
        
    def one_hot_encode(self, y, num_classes, activation_function):
        encoded = np.zeros((len(y), num_classes))
        for i, label in enumerate(y):
            if activation_function == 'logística':
                encoded[i, label - 1] = 1  # One-hot encoding
            elif activation_function == 'tangente':
                encoded[i] = -1  # Start with all -1
                encoded[i, label - 1] = 1  # Bipolar encoding
        return encoded

    def test(self, X):
        output = self.forward(X)
        if self.activation_function == np.tanh:
            return np.where(output > 0, 1, -1)
        else:
            return np.argmax(output, axis=1)

