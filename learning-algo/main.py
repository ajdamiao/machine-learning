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
        k = neuron.weight[i] * x[i]
        
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
    
if __name__ == "__main__":
    neuron = initneuron(ident, 4)
    
    out_true: float = []
    x = [[0], [2], [4], [6]]
    
    out_true : float = [6, 11, 16, 21]
    out_pred: list = []
    
    for i in range(0,4):
        out_pred.append(computout(neuron, x[i]))
        
    print('O valor de w é ', neuron.weight[0])
    print('O Valor do bias é ', neuron.bias)
    
    print('O custo do neuronio é: ', mse(out_true, out_pred, 4))
    