# My implementation

# Redes Neurais Artificiais em Python
[Udemy](https://www.udemy.com)

# Perceptron

- Files
    - perceptron-1.0.py
    - perceptron-1.1.py
    - perceptron-1.2.py

# Perceptron Multilayer

- Files
    - perceptron-multilayer-1.0.py

        - simple example of a multi-layered neural network using predefined input data.

    - perceptron-multilayer-1.1.py

        - example of a multi-layered neural network using real input data.

        - Dataset Test: [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). Não é necessário baixar o dataset, pois já é nativa no [sklearn](http://scikit-learn.org/) do python.

# Conceptualization 

- Table XOR

    | X1 | X2 | Class |
    |:-: | :-:|  :-:  |
    | 0 | XOR | 0 | 0 | 
    | 0 | XOR | 1 | 1 |
    | 1 | XOR | 0 | 1 |
    | 1 | XOR | 1 | 0 |

- Backpropagation
    
    peso n+1 = (poso n * momento) + (entrada * delta * taxa de aprendizagem)

- Error

    - Mean Square Error (MSE)
    - Root Mean Square Error (RMSE)

- Cross Validation (Validação Cruzada)

# Links
[Activation function](https://en.wikipedia.org/wiki/Activation_function)

[Numpy Exp](http://pyscience-brasil.wikidot.com/docitem:numpy-exp)

[Vectors](https://www.mathsisfun.com/algebra/vectors.html)

[MPLClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

[Outros cursos](http://iaexpert.com.br/index.php/cursos/)