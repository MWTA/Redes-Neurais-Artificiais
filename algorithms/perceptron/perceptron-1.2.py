'''
    @Software: Perceptron
    @Description: Algorítimo Perceptron Melhoria de Desempenho para trabalhar com muitos dados
                  com controle de erros (calculo de pesos) e treinamento. (Function Step Degrau)
    @Date: 08/05/2018
'''
#%%
import numpy as np

_input = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)

_output = np.array(
    [0, 0, 0, 1]
)

_weights = np.array([0.0, 0.0])

_learning_rate = 0.1


def stepFunction(_soma):
    if(_soma >= 1):
        return 1
    return 0


def calc_output(_register):
    # dot protuct / produto escalar.
    # melhora o desempenho para muitos dados.
    return stepFunction(_register.dot(_weights))


def training():
    total_error = 1

    while(total_error != 0):

        total_error = 0

        for i in range(len(_output)):
            calculated_output = calc_output(np.asarray(_input[i]))
            error = abs(_output[i] - calculated_output)
            total_error += error
            for j in range(len(_weights)):
                _weights[j] = _weights[j] + (_learning_rate * _input[i][j] * error)
                print("Peso atualizado: " + str(_weights[j]))

        print("Total de erros: " + str(total_error))


print(training())
print("Rede neural treinada")
print(calc_output(_input[0]))
print(calc_output(_input[1]))
print(calc_output(_input[2]))
print(calc_output(_input[3]))
