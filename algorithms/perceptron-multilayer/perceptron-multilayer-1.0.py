# -*- coding: utf-8 -*-
'''
    @Software: Perceptron Multilayer
    @Description: Algorítimo Perceptron Multicamadas (Function Step Sigmoid).
    @Date: 08/05/2018
'''
#%%
import numpy as np


def sigmoid(_synapse):
    return 1 / (1 + np.exp(-_synapse))


def sigmoid_derivative(_sig):
    return _sig * (1 - _sig)


input_value = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

output_layer = np.array([
    [0], [1], [1], [0]
])

'''
weights_input = np.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469]
])

weights_hidden = np.array([
    [-0.017],
    [-0.893],
    [0.148]
])
'''

# Define pesos aleatórios, 2 neurónio na camada de entrada e 3 na camada escondiada.
weights_input = 2 * np.random.random((2, 3)) - 1

# Define pesos aleatórios, 3 na camada escondida e  na camada de saída.
weights_hidden = 2 * np.random.random((3, 1)) - 1

epoch = 10
learning_rate = 0.5
moment = 1

for j in range(epoch):
    # Entrada de valores.
    layer_input = input_value
    
    # SOMA - Sinapse de entrada com os pesos e valores. 
    synapse_input = np.dot(layer_input, weights_input)
    layer_hidder = sigmoid(synapse_input)

    # DERIVADA -
    synapse_exit = np.dot(layer_hidder, weights_hidden)
    layer_exit = sigmoid(synapse_exit)

    # ERRO - 
    error_layer_exit = output_layer - layer_exit
    absolute_average = np.mean(np.abs(error_layer_exit))
    print("Erro: " + str(absolute_average))

    # DELTA SAÌDA - Calcula o Delta para a camada de saída saída.
    derivative_exit = sigmoid_derivative(layer_exit)
    delta_exit = error_layer_exit * derivative_exit

    # DELTA ESCONDIDA - Calcula o Delta para a camada escondida.
    weights_transposed = weights_hidden.T
    delta_exit_weights = delta_exit.dot(weights_transposed)
    delta_layer_hidder = delta_exit_weights * sigmoid_derivative(layer_hidder)

    # Backpropagation
    layer_hidder_transposed = layer_hidder.T
    new_weights_hidder = layer_hidder_transposed.dot(delta_exit)
    weights_hidden = (weights_hidden * moment) + (new_weights_hidder * learning_rate)

    layer_input_transposed = layer_input.T
    new_weights_input = layer_input_transposed.dot(delta_layer_hidder)
    weights_input = (weights_input * moment) + (new_weights_input * learning_rate)



