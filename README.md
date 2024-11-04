# Projeto de Inteligência Artificial

Este projeto implementa uma rede neural para classificação, com funcionalidades para treinamento e teste, além de exibir a matriz de confusão.

## Estrutura do Código

`src/main.py`

- Ponto de entrada do programa. Gerencia a interação com o usuário e coordena as diferentes partes do programa.

`src/utils.py`

- Contém funções utilitárias para carregar dados e calcular o número de neurônios na camada oculta.

`src/neural_network.py`

- Implementa a classe `NeuralNetwork` que define a estrutura e os métodos da rede neural.

`src/metrics.py`

- Contém a função para calcular a matriz de confusão. 

## Funcionalidades

1. Informar o nome do arquivo de treinamento.
2. Abrir o arquivo e ver quantos atributos e quantas classes existem.
3. Indicar o número de neurônios na camada de entrada e saída.
4. Calcular o número de neurônios na camada oculta (média geométrica), permitindo que o usuário o altere.
5. Informar a função de transferência (logística ou tangente hiperbólica).
6. Informar a condição de parada (número de iterações ou erro máximo permitido).
7. Criar a rede neural (matrizes contendo os pesos iniciais).
8. Treinar a rede neural até satisfazer a condição de parada.
9. Informar o nome do arquivo de teste.
10. Realizar o teste.
11. Apresentar a matriz de confusão.


## Configuração do Ambiente
### Pré-requisitos

- Python 3.6 ou superior
- pip (gerenciador de pacotes do Python)

### 1. Clonar o Repositório

```sh
git clone https://github.com/mnazaro/bacpropagation_neuralnet.git
cd backpropagation_neuralnet
```
### 2. Criar e Ativar o Ambiente Virtual
**Windows**
```sh
python -m venv venv
venv\Scripts\activate
```
**Linux**
```sh
python3 -m venv venv
source venv/bin/activate
```
### 3. Instalar as dependências
```sh
pip install -r requirements.txt
```
### 4. Configurar variáveis de ambiente
Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:
```
TRAINING_FILE=../data/treinamento.csv
TEST_FILE=../data/teste.csv
```
## Executar o projeto
**Windows**
Renomeie o arquivo `start.windows` para `start.bat`
```
start.bat
```
**Linux**
Renomeie o arquivo `start.linux` para `start.sh`
```
start.sh
```

