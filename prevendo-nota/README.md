# Rede neural simples (prevendo-nota)

Este projeto implementa um neuronio unico (regressao linear) treinado por gradiente numerico (diferencas finitas). O objetivo e ajustar pesos e bias para aproximar saidas reais a partir de entradas.

## Estrutura geral do codigo

O arquivo `main.py` tem:

- Um modelo `Neuron` que guarda:
  - `weight`: lista de pesos (um para cada entrada).
  - `nconnections`: quantidade de entradas (tamanho de `weight`).
  - `bias`: termo independente.
  - `actfunc`: funcao de ativacao (aqui e identidade).
- Funcoes de utilidade para:
  - inicializar o neuronio com pesos aleatorios;
  - calcular a saida do neuronio;
  - calcular o custo (erro);
  - calcular o gradiente numerico;
  - treinar (atualizar pesos e bias).
- Um bloco `if __name__ == "__main__":` que cria dados, imprime valores antes e depois do treino e executa o treinamento.

## Modelo de neuronio

A classe `Neuron` representa um unico neuronio. Ele calcula:

```
saida = actfunc(soma(peso_i * entrada_i) + bias)
```

Como a ativacao e identidade, o resultado final e apenas a soma linear.

## Funcao de ativacao (identidade)

A funcao `ident(x)` retorna `x`. Isso significa que o neuronio e linear, sem nao-linearidade.

## Inicializacao (`initneuron`)

- Cria um neuronio com `nconnections` pesos.
- Cada peso e o bias recebem valores aleatorios entre -1 e 1.
- Isso evita iniciar tudo em zero (o que impediria o treino).

## Saida do neuronio (`computout`)

- Para uma entrada `x` (lista com tamanho igual ao numero de pesos), calcula:

```
k = sum(weight[i] * x[i]) + bias
```

- Retorna `actfunc(k)`.

## Funcao de custo (MSE)

A funcao `mse` calcula o erro medio quadratico:

```
MSE = (1/n) * sum((y_pred[i] - y_true[i])^2)
```

Onde:
- `y_true`: saidas reais.
- `y_pred`: saidas preditas pelo neuronio.
- `n`: numero de amostras.

## Custo total (`comput_cost`)

- Para cada amostra de entrada, gera uma predicao com `computout`.
- Junta todas as predicoes em `out_pred`.
- Aplica a funcao de custo (`mse`).

Isso retorna um unico valor: o custo total do neuronio nas amostras.

## Gradiente numerico (`comput_gradient`)

Como nao existe derivada analitica implementada, o codigo usa **diferencas finitas**:

```
(dC/dp) ≈ (C(p + epsilon) - C(p)) / epsilon
```

Onde:
- `p` e um parametro (peso ou bias).
- `epsilon` e um valor pequeno.

No codigo:

1. Guarda o valor original do peso ou bias.
2. Soma `epsilon` ao parametro.
3. Calcula o custo (`variation_cost`).
4. Restaura o valor original.
5. Calcula o custo normal (`normal_cost`).
6. Calcula o gradiente pela formula acima.

Isso e feito para cada peso e para o bias.

## Treinamento (`train`)

O treino faz **descida do gradiente**:

```
parametro = parametro - lr * gradiente
```

- Para cada peso:
  - Calcula o gradiente numerico.
  - Atualiza o peso.
- Depois atualiza o bias da mesma forma.

`lr` (learning rate) controla o tamanho do passo.

## Dados de treino

No bloco principal:

- `x` e uma lista de amostras.
  - Cada amostra e uma lista com valores de entrada.
  - O numero de colunas de `x` deve ser igual ao numero de pesos.
- `out_true` sao as saidas esperadas.

Exemplo:

```
x = [
  [1, 5],
  [2, 8],
  ...
]
```

Isso significa 2 features por amostra. Logo, o neuronio deve ter 2 pesos.

## Por que so alguns pesos mudam

- Se uma coluna de `x` for sempre zero, o peso correspondente nao participa da soma.
- Nesse caso, o gradiente desse peso vira zero, e ele nao muda no treino.

Por isso, o numero de pesos deve ser igual ao numero de entradas por amostra, e as colunas precisam ter valores reais.

## Fluxo completo de execucao

1. Inicializa neuronio com pesos e bias aleatorios.
2. Imprime pesos, bias e custo inicial.
3. Executa o treino por N epocas (loop).
4. Imprime pesos, bias e custo final.

Se o treino estiver configurado corretamente, o custo deve diminuir.

## Dicas de ajuste

- Se o custo explodir, diminua o `learning rate`.
- Se o treino estiver lento, aumente `learning rate` com cuidado.
- Garanta que `sample_size` seja igual ao numero de amostras em `x` e `out_true`.
- Se precisar de mais estabilidade, normalize os valores de entrada e saida.
