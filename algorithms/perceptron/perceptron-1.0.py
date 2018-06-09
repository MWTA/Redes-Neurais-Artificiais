'''
    @Software: Perceptron
    @Description: Algoritimo Perceptron básico. (Function Step Degrau)
    @Date: 08/05/2018
'''

#%%

_input = [-1, 7, 5]
_weight = [0.8, 0.1, 0]

def calc_soma(_input, _weight):
    s = 0
    for i in range(3):
        s += _input[i] * _weight[i]
    return s

s = calc_soma(_input, _weight)
print(s)

def stepFunction(_soma):
    if(s >= 1):
        return 1
    return 0

r = stepFunction(s)
print(r)
