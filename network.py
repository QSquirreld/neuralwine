from __future__ import annotations
from typing import List, Callable, TypeVar, Tuple
from functools import reduce
from layer import Layer
from util import sigmoid, derivative_sigmoid

T = TypeVar('T')  # тип вывода интерпретации нейронной сети


class Network:
    def __init__(self, layer_structure: List[int], learning_rate: float,
                 activation_function: Callable[[float], float] = sigmoid,
                 derivative_activation_function: Callable[[float], float] = derivative_sigmoid) -> None:
        if len(layer_structure) < 3:
            raise ValueError("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)")
        self.layers: List[Layer] = []
        # слой ввода
        input_layer: Layer = Layer(None, layer_structure[0], learning_rate, activation_function,
                                   derivative_activation_function)
        self.layers.append(input_layer)
        # скрытые слои и выходной слой
        for previous, num_neurons in enumerate(layer_structure[1::]):
            next_layer = Layer(self.layers[previous], num_neurons, learning_rate, activation_function,
                               derivative_activation_function)
            self.layers.append(next_layer)

    # Отправляет входные данные на первый уровень, а затем выводит из первого в качестве входных данных для второго,
    # второго для третьего и т.д.
    def outputs(self, input: List[float]) -> List[float]:
        return reduce(lambda inputs, layer: layer.outputs(inputs), self.layers, input)

    # Вычислите изменения каждого нейрона на основе ошибок вывода в сравнении с ожидаемым результатом
    def backpropagate(self, expected: List[float]) -> None:
        # вычислить дельту для нейронов выходного слоя
        last_layer: int = len(self.layers) - 1
        self.layers[last_layer].calculate_deltas_for_output_layer(expected)
        # вычислить дельту для скрытых слоев в обратном порядке
        for l in range(last_layer - 1, 0, -1):
            self.layers[l].calculate_deltas_for_hidden_layer(self.layers[l + 1])

    # backpropagate() на самом деле не изменяет никаких весов
    # эта функция использует дельты, вычисленные в backpropagate(), чтобы
    # на самом деле внесите изменения в веса
    def update_weights(self) -> None:
        for layer in self.layers[1:]: # пропустить входной слой
            for neuron in layer.neurons:
                for w in range(len(neuron.weights)):
                    neuron.weights[w] = neuron.weights[w] + (neuron.learning_rate * (layer.previous_layer.output_cache[w]) * neuron.delta)

    # train() использует результаты outputs(), которые выполняются по многим входам и сравниваются
    # против ожидаемых значений для подачи backpropagate() и update_weights()
    def train(self, inputs: List[List[float]], expecteds: List[List[float]]) -> None:
        for location, xs in enumerate(inputs):
            ys: List[float] = expecteds[location]
            outs: List[float] = self.outputs(xs)
            self.backpropagate(ys)
            self.update_weights()

    # для обобщенных результатов, требующих классификации, эта функция вернет
    # правильное количество попыток и правильный процент от общего числа
    def validate(self, inputs: List[List[float]], expecteds: List[T],
                 interpret_output: Callable[[List[float]], T]) -> Tuple[int, int, float]:
        correct: int = 0
        for input, expected in zip(inputs, expecteds):
            result: T = interpret_output(self.outputs(input))
            if result == expected:
                correct += 1
        percentage: float = correct / len(inputs)
        return correct, len(inputs), percentage
