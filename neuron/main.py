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

if __name__ == "__main__":
    
    neuron = initneuron(ident, 4)
    
    x = [10, 6, -8, 5]
    
    print('A saida do neuronio é: ', computout(neuron, x=x))
    