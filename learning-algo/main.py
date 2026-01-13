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
    neuron = initneuron(ident, 4)
    
    out_true: list[float] = []
    
    
    # define entradas
    x = [
        [0,0],
        [2,0],
        [4,0],
        [6,0],
        [6,0],
        [6,0],
        [6,0],
    ]
    

    
    # Saida verdadeira
    out_true : float = [6, 11, 16, 21]
    
    # Saida predita
    out_pred: list = []
    
    neuron.weight[0] = 2.5
    neuron.bias = 1
        
    print('O valor de weight é ', neuron.weight[0])
    print('O Valor do bias é ', neuron.bias)
    print('O custo do neuronio é: ', comput_cost(neuron, x, out_true, mse, 4))
    
    for i in range(0, 100000):
        train(neuron, mse, x, out_true, 4)
        
    print('')
    print('O valor de WEIGHT depois do treino é ', neuron.weight[0])
    print('O Valor do BIAS depois do treino  é ', neuron.bias)
    print('O CUSTO do neuronio dps do treino : ', comput_cost(neuron, x, out_true, mse, 4))
        
    
