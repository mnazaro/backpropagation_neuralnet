import os
from dotenv import load_dotenv
from utils import load_data, calculate_hidden_neurons

load_dotenv()

def main():
    training_file = os.getenv('TRAINING_FILE')
    test_file = os.getenv('TEST_FILE')

    data, num_attributes, num_classes = load_data(training_file)
    input_neurons = num_attributes
    output_neurons = num_classes
    hidden_neurons = calculate_hidden_neurons(input_neurons, output_neurons)
    
    print(f'Input neurons: {input_neurons}')
    print(f'Output neurons: {output_neurons}')
    print(f'Hidden neurons: {hidden_neurons}')

if __name__ == '__main__':
    main()