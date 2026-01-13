from typing import Callable
from pydantic import BaseModel
import random

# model do neuronio
class Neuron(BaseModel):
    weight: list
    nconnections: int
    bias: float
    actfunc: Callable[[float], float]
    
# funcao de ativacao identidade
def ident(x: float) -> float:
    return x

def computout(neuron: Neuron, x: list):
    k = 0

    for i in range(len(x)):
        k += neuron.weight[i] * x[i]

    k = k + neuron.bias

    return neuron.actfunc(k)


def randomize(min: float, max:float):
    return random.uniform(min, max)
    
# inicializa um neuronio
def initneuron(actfunc: Callable[[float], float], nconnections: int) -> Neuron:
    neuron = Neuron(weight=[], nconnections=nconnections, bias=0.0, actfunc=actfunc)

    for i in range(nconnections):
        neuron.weight.append(randomize(-1,1))
        
    neuron.bias = randomize(-1,1)
    
    return neuron

# Funcao de custo
def mse(out_true: list, out_pred: list, samplesize: int) -> float:
    
    s : float = 0
    for i in range(samplesize):
        s += pow(out_pred[i] - out_true[i], 2)
    
    s /= samplesize
    return s        

# x = entrada verdadeira. | y = saidas verdadeiras
# funcao de custo
def comput_cost(neuron: Neuron, x: list[list[int]], y: list[float], cost: Callable[[list[float], list[float], int], float], sample_size: int):
    out_pred: list[float] = []
    for i in range(0, sample_size):
        out_pred.append(computout(neuron, x[i]))

    return cost(y, out_pred, sample_size)
    
    
# x = entrada verdadeira. | y = saidas verdadeiras
# computar gradiente
def comput_gradient(neuron: Neuron, cost: Callable[[list[float], list[float], int], float], x: list[list[int]], y: list[float], param_ref: tuple[str, int | None], sample_size: int):
    #                   limite funcao cost(param + delta param) - cost(param)
    # delta param -> 0 ------------------------------------------------------------  divido
    #                                       delta param
    # delta param + param
    
    ref_type, ref_index = param_ref
    if ref_type == "weight" and ref_index is not None:
        original = neuron.weight[ref_index]
        neuron.weight[ref_index] = original + 0.0001
        variation_cost: float = comput_cost(neuron, x, y, cost, sample_size)
        neuron.weight[ref_index] = original
        normal_cost = comput_cost(neuron, x, y, cost, sample_size)
    else:
        original = neuron.bias
        neuron.bias = original + 0.0001
        variation_cost = comput_cost(neuron, x, y, cost, sample_size)
        neuron.bias = original
        normal_cost = comput_cost(neuron, x, y, cost, sample_size)
    
    gradient = (variation_cost - normal_cost) / 0.0001
    
    return gradient

# x = entrada verdadeira. | y = saidas verdadeiras
def train(neuron: Neuron, cost: Callable[[list[float], list[float], int], float], x: list[list[int]], y: list[float], sample_size: int):
    gradient: float
    
    for i in range(0, neuron.nconnections):
        gradient = comput_gradient(neuron, cost, x, y, ("weight", i), sample_size)
        
        neuron.weight[i] -= 0.001 * gradient
    
    gradient = comput_gradient(neuron, cost, x, y, ("bias", None), sample_size)
    neuron.bias -= 0.001 * gradient
    
    
if __name__ == "__main__":
    neuron = initneuron(ident, 2)
    
    out_true: list[float] = []
    
    
    # define entradas
    x = [[0, 0] for _ in range(6)]

    # horas de estudos
    x[0][0] = 1
    x[1][0] = 2
    x[2][0] = 4
    x[3][0] = 5
    x[4][0] = 9
    x[5][0] = 8
    
    # horas de sono
    x[0][1] = 5
    x[1][1] = 8
    x[2][1] = 6
    x[3][1] = 9
    x[4][1] = 8
    x[5][1] = 5
    
    
    # Saida verdadeira (nota da prova)
    out_true : float = [ 3.2, 4.5, 5, 6.8, 8.2, 6 ]
    
    # Saida predita
    out_pred: list = []
    
    #neuron.weight[0] = 2.5
    #neuron.bias = 1
        
    print("Pesos antes do treino:", neuron.weight[0])
    print("Bias antes do treino:", neuron.bias)
    print("Custo antes do treino:", comput_cost(neuron, x, out_true, mse, 6))
    
    for i in range(0, 50000):
        train(neuron, mse, x, out_true, 6)
        
    print("")
    print("Pesos depois do treino:", neuron.weight[0])
    print("Bias depois do treino:", neuron.bias)
    print("Custo depois do treino:", comput_cost(neuron, x, out_true, mse, 6))
    
    print(" ")
    
    for i in range(0, 6):
        print(f'entrada  {x[i][0]} {computout(neuron, x[i])}')
        
    
