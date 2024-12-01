import os
import numpy as np
from dotenv import load_dotenv
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QLabel, QLineEdit, QRadioButton,
    QButtonGroup, QPushButton, QWidget, QMessageBox, QHBoxLayout,
    QTextEdit
)

import utils
import metrics
import neural_network

load_dotenv()
TRAINING_FILE = os.getenv("TRAINING_FILE")
TEST_FILE = os.getenv("TEST_FILE")

class NeuralNetworkApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rede Neural Backpropagation")
        self.init_ui()

    def init_ui(self):
        container = QHBoxLayout()
        layout_right = QVBoxLayout()
        layout_left = QVBoxLayout()

        self.training_file_label = QLabel("Arquivo de Treinamento:")
        self.training_file_input = QLineEdit()
        self.training_file_input.setPlaceholderText("Digite o nome do arquivo de treinamento")

        self.test_file_label = QLabel("Arquivo de Teste:")
        self.test_file_input = QLineEdit()
        self.test_file_input.setPlaceholderText("Digite o nome do arquivo de teste")

        self.input_label = QLabel("Camada de entrada: Calculando...")
        self.output_label = QLabel("Camada de saída: Calculando...")

        self.hidden_label = QLabel("Camada oculta (média geométrica):")
        self.hidden_input = QLineEdit()
        self.hidden_input.setPlaceholderText("Digite o número de neurônios na camada oculta")

        self.activation_label = QLabel("Função de ativação:")
        self.activation_group = QButtonGroup(self)
        self.activation_logistic = QRadioButton("Logística")
        self.activation_tanh = QRadioButton("Tangente Hiperbólica")
        self.activation_logistic.setChecked(True)
        self.activation_group.addButton(self.activation_logistic)
        self.activation_group.addButton(self.activation_tanh)

        activation_layout = QHBoxLayout()
        activation_layout.addWidget(self.activation_logistic)
        activation_layout.addWidget(self.activation_tanh)

        self.criteria_label = QLabel("Condição de parada:")
        self.criteria_group = QButtonGroup(self)
        self.criteria_error = QRadioButton("Erro máximo")
        self.criteria_iterations = QRadioButton("Número de iterações")
        self.criteria_error.setChecked(True)
        self.criteria_group.addButton(self.criteria_error)
        self.criteria_group.addButton(self.criteria_iterations)

        criteria_layout = QHBoxLayout()
        criteria_layout.addWidget(self.criteria_error)
        criteria_layout.addWidget(self.criteria_iterations)

        self.criteria_input = QLineEdit()
        self.criteria_input.setPlaceholderText("Digite o valor para o critério escolhido...")

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setPlaceholderText("Mensagens do sistema aparecerão aqui...")

        self.train_button = QPushButton("Treinar Rede Neural")
        self.train_button.clicked.connect(self.train_network)
        self.test_button = QPushButton("Testar Rede Neural")
        self.test_button.clicked.connect(self.test_network)

        layout_left.addWidget(self.training_file_label)
        layout_left.addWidget(self.training_file_input)
        layout_left.addWidget(self.test_file_label)
        layout_left.addWidget(self.test_file_input)
        layout_left.addWidget(self.input_label)
        layout_left.addWidget(self.output_label)
        layout_left.addWidget(self.hidden_label)
        layout_left.addWidget(self.hidden_input)
        layout_left.addWidget(self.activation_label)
        layout_left.addLayout(activation_layout)
        layout_left.addWidget(self.criteria_label)
        layout_left.addLayout(criteria_layout)
        layout_left.addWidget(self.criteria_input)
        layout_left.addWidget(self.train_button)
        layout_left.addWidget(self.test_button)
        layout_right.addWidget(QLabel("Log:"))
        layout_right.addWidget(self.log_window)

        container.addLayout(layout_left)
        container.addLayout(layout_right)
        self.setLayout(container)
        self.load_initial_values()

    def load_initial_values(self):
        try:
            training_file = self.training_file_input.text() or TRAINING_FILE
            test_file = self.test_file_input.text() or TEST_FILE

            X, y = utils.load_data(training_file)
            input_neurons = X.shape[1]
            output_neurons = len(set(y))
            
            hidden_neurons = utils.calculate_hidden_neurons(input_neurons, output_neurons)

            self.input_label.setText(f"Camada de entrada: {input_neurons}")
            self.output_label.setText(f"Camada de saída: {output_neurons}")
            self.hidden_input.setText(str(hidden_neurons))

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao carregar os arquivos: {str(e)}")

    def log_message(self, message):
        self.log_window.append(message)

    def train_network(self):
        try:
            activation_function = "logística" if self.activation_logistic.isChecked() else "tangente"
            output_neurons = int(self.output_label.text().split(":")[-1].strip())
            X, y = utils.load_data(TRAINING_FILE)
            y_encoded = utils.encode_labels(y, output_neurons, activation_function)
            input_neurons = int(self.input_label.text().split(":")[-1].strip())
            hidden_neurons = int(self.hidden_input.text())
            
            nn = neural_network.NeuralNetwork(input_neurons, hidden_neurons, output_neurons, activation_function)

            if self.criteria_error.isChecked():
                error_threshold = float(self.criteria_input.text())
                max_epochs = 5000
            else:
                max_epochs = int(self.criteria_input.text())
                error_threshold = None

            self.log_message(f"Iniciando treinamento com critérios: {max_epochs} épocas e erro máximo de {error_threshold}")
            nn.train(X, y_encoded, max_epochs=max_epochs, error_threshold=error_threshold, learning_rate=0.01,
                     log_callback=self.log_message)
            QMessageBox.information(self, "Treinamento", "Treinamento concluído com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao treinar a rede neural: {str(e)}")

    def test_network(self):
        try:
            X_test, y_test = utils.load_data(TEST_FILE)
            activation_function = "logística" if self.activation_logistic.isChecked() else "tangente"
            output_neurons = int(self.output_label.text().split(":")[-1].strip())
            input_neurons = int(self.input_label.text().split(":")[-1].strip())
            hidden_neurons = int(self.hidden_input.text())

            nn = neural_network.NeuralNetwork(input_neurons, hidden_neurons, output_neurons, activation_function)

            predictions = nn.test(X_test)

            if activation_function == "tangente":
                predictions = np.where(predictions > 0, 1, -1) 

            metrics.generate_confusion_matrix(y_test, predictions, log_callback=self.log_message)
            QMessageBox.information(self, "Teste", "Teste concluído com sucesso!")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao testar a rede neural: {str(e)}")
    
if __name__ == "__main__":
    app = QApplication([])
    window = NeuralNetworkApp()
    window.show()
    app.exec()
