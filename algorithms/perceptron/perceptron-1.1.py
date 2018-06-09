'''
    @Software: Perceptron
    @Description: Algorítimo Perceptron Melhoria de Desempenho para trabalhar com 
                  muitos dados. (Function Step Degrau)
    @Date: 08/05/2018
'''
#%%
import numpy as np

_input = np.array([-1, 7, 5])
_weight = np.array([0.8, 0.1, 0])

def calc_soma(_input, _weight):
    # dot protuct / produto escalar.
    # melhora o desempenho para muitos dados.
    return _input.dot(_weight)

s = calc_soma(_input, _weight)
print(s)

def stepFunction(_soma):
    if(s >= 1):
        return 1
    return 0

r = stepFunction(s)
print(r)
