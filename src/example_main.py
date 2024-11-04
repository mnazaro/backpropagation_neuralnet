def main():
    # 1. Informar o nome do arquivo de treinamento
    training_file = input("Informe o nome do arquivo de treinamento: ")

    # 2. Abrir o arquivo e ver quantos atributos e quantas classes existem
    data, num_attributes, num_classes = load_data(training_file)
    
    # 3. Indicar o número de neurônios na camada de entrada e saída
    input_neurons = num_attributes
    output_neurons = num_classes
    print(f"Número de neurônios na camada de entrada: {input_neurons}")
    print(f"Número de neurônios na camada de saída: {output_neurons}")

    # 4. Calcular o número de neurônios na camada oculta
    hidden_neurons = calculate_hidden_neurons(input_neurons, output_neurons)
    user_hidden_neurons = input(f"Número de neurônios na camada oculta (sugerido: {hidden_neurons}): ")
    hidden_neurons = int(user_hidden_neurons) if user_hidden_neurons else hidden_neurons

    # 5. Informar a função de transferência
    transfer_function = input("Informe a função de transferência (logística ou tangente hiperbólica): ").strip().lower()

    # 6. Informar a condição de parada
    stop_condition = input("Informe a condição de parada (iterações ou erro máximo): ").strip().lower()
    if stop_condition == "iterações":
        max_iterations = int(input("Informe o número máximo de iterações: "))
        stop_value = max_iterations
    elif stop_condition == "erro máximo":
        max_error = float(input("Informe o erro máximo permitido: "))
        stop_value = max_error
    else:
        raise ValueError("Condição de parada inválida")

    # 7. Criar a rede neural
    nn = NeuralNetwork(input_neurons, hidden_neurons, output_neurons, transfer_function)

    # 8. Treinar a rede neural
    nn.train(data, stop_condition, stop_value)
    print("Treinamento concluído")

    # 9. Informar o nome do arquivo de teste
    test_file = input("Informe o nome do arquivo de teste: ")
    test_data, _, _ = load_data(test_file)

    # 10. Realizar o teste
    predictions = nn.test(test_data)

    # 11. Apresentar a matriz de confusão
    cm = confusion_matrix(test_data[:, -1], predictions)
    print("Matriz de Confusão:")
    print(cm)

if __name__ == "__main__":
    main()