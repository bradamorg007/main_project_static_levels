import numpy as np
from old_code.Random import Random
import math
import os
import pickle as pkl
import bz2

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    # leaky
    if x < 0:
        return 0.01
    else:
        return x


def tanh(x):
    return math.tanh(x)


def softmax(x):

    max = np.max(x)

    # correct data by shifting values below 0 to stop softmax becoming unstable
    x = np.subtract(x, max)

    exps = np.vectorize(math.exp)
    exps = exps(x)

    sum_exps = np.sum(exps)

    return np.divide(exps, sum_exps)



class NeuralNetwork:

    def __init__(self, layer_units=None, activation_func_list=None):

        if layer_units == None or activation_func_list == None:

            self.num_layers = None
            self.layers = None
            self.layer_units = None
            self.weight_init = None
            self.activation_functions = None

        else:

            self.num_layers = len(layer_units)
            self.layers = [{}] * (len(layer_units) -1)
            self.layer_units = layer_units
            self.weight_init = False
            self.activation_functions = activation_func_list

        self.CLASS_TAG = "nn.py"
        self.name = 'train_agent'


    def init_layers(self, init_type=None):

        error = False
        for unit in self.layer_units:
            if unit < 1:
                error = True
                break

        if error or len(self.layer_units) == 0:
            raise ValueError(
                "Illegal Argument Exception: the topology of the network has not been defined")

        for i in range(1, len(self.layer_units)):

            current_layer = self.layer_units[i]
            previous_layer = self.layer_units[i-1]

            str_current = "H" + str(i)
            str_previous = "H" + str(i - 1)

            if i == 1:
                str_previous = "in" + str(i - 1)

            if i == len(self.layer_units) - 1:
                str_current = "out" + str(i)

            current_activation_function = None
            if len(self.activation_functions) == 1:
                current_activation_function = self.activation_functions[0]
            else:
                current_activation_function = self.activation_functions[i - 1]

            Layer_to_layer = {
                "tag": str_previous + "-" + str_current,
                "weights": np.zeros(shape=(current_layer, previous_layer)),
                "biases": np.zeros(shape=(current_layer, 1)),
                "activation_function": current_activation_function
            }

            if init_type == "normal":
                Layer_to_layer['weights'] = Random.normal(Layer_to_layer.get('weights'))
                Layer_to_layer['biases'] = Random.normal(Layer_to_layer.get('biases'))

            elif init_type == "he_normal":
                Layer_to_layer['weights'] = Random.he_normal(Layer_to_layer.get('weights'))
                Layer_to_layer['biases'] = Random.he_normal(Layer_to_layer.get('biases'))

            elif init_type == "xavier":
                Layer_to_layer['weights'] = Random.xavier_normal(Layer_to_layer.get('weights'))
                Layer_to_layer['biases'] = Random.xavier_normal(Layer_to_layer.get('biases'))

            elif init_type == "truncated_normal":
                Layer_to_layer['weights'] = Random.truncated_normal(Layer_to_layer.get('weights'))
                Layer_to_layer['biases'] = Random.truncated_normal(Layer_to_layer.get('biases'))

            elif init_type == "uniform_random":
                Layer_to_layer['weights'] = Random.uniform_random(Layer_to_layer.get('weights'))
                Layer_to_layer['biases'] = Random.uniform_random(Layer_to_layer.get('biases'))

            elif init_type == "zeros":
                pass

            else:
                raise ValueError(
                    "Illegal Argument Exception: non valid init_type detected")


            self.layers[i - 1] = Layer_to_layer

        self.weight_init = True


    def feed_foward(self, inputs):

        if not isinstance(inputs, np.ndarray):
            raise ValueError("Illegal Argument Exception: inputs into the network must be a numpy array")

        if len(inputs.shape) <= 1:
            raise ValueError("Error: numpy 0 dimension detected, please use np.expand_dims on the zero axis to"+
                             "to make the np array shape a tuple e.g (1, 5) ")

        inputs = np.reshape(inputs, newshape=(inputs.size, 1))

        for layer in self.layers:

            inputs = np.dot(layer.get('weights'), inputs)
            inputs = np.add(inputs, layer.get('biases'))

            if layer.get('activation_function') == 'sigmoid':
                func = np.vectorize(sigmoid)
                inputs = func(inputs)

            elif layer.get('activation_function') == 'relu':
                func = np.vectorize(relu)
                inputs = func(inputs)

            elif layer.get('activation_function') == 'tanh':
                func = np.vectorize(tanh)
                inputs = func(inputs)

            elif layer.get('activation_function') == 'softmax':
                inputs = softmax(inputs)

        return inputs


    def mutate(self, rate):

        def update(x):

            if np.random.rand() < rate:
                x = x + Random.gaussian_distribution(mean=0, sigma=1, samples=1)
                return x

            return x

        update = np.vectorize(update)
        for i, layer in enumerate(self.layers):

            self.layers[i]['weights'] = update(layer['weights'])
            self.layers[i]['biases'] = update(layer['biases'])


    @staticmethod
    def standardization(input):

        if not isinstance(input, np.ndarray):
            raise ValueError("Illegal Argument Exception: inputs into the network must be a numpy array")

        mean = np.mean(input)
        stdev = np.std(input)

        if stdev != 0:
            func = lambda x: (x - mean) / stdev
            func = np.vectorize(func)
            input = func(input)

        return input

    @staticmethod
    def normalise(input):

        if not isinstance(input, np.ndarray):
            raise ValueError("Illegal Argument Exception: inputs into the network must be a numpy array")

        max = np.max(input)
        min = np.min(input)

        s = input.shape
        rows = s[0]
        cols = s[1]

        for i in range(rows):
            for j in range(cols):
                x = input[i][j]
                input[i][j] = (x - min) / (max - min)


        # func = lambda x : (x - min) / (max - min)
        # func = np.vectorize(func)
        # input = func(input)

        return input


    def save_model(self, filename, folderpath="../models/"):

        if os.path.isdir(folderpath) == False:
            os.mkdir(folderpath)

        fullpath = bz2.BZ2File(os.path.join(folderpath, filename), 'w')
        pkl.dump(self, fullpath)
        fullpath.close()

        print("Neural Network Save Complete: %s" % filename)


    @staticmethod
    def load_model(filename):

        if os.path.exists(filename) == False:
            raise ValueError("Error Neural Network Load_model: Filename does not exist")

        fullpath = bz2.open(filename, 'r')
        model = pkl.load(fullpath)
        fullpath.close()

        file_tag = model.CLASS_TAG

        if file_tag == None or file_tag != 'nn.py':
            raise ValueError("ERROR Load_model: The loaded file is not a valid NeuralNetwork.py object")

        return model


if __name__ == "__main__":
    pass
    # this is just some test function
    # import matplotlib.pyplot as plt
    #
    # #net = NeuralNetwork.load_model("../models/test_net")
    #
    # net = NeuralNetwork(layer_units=[5, 500, 4, 100, 10], activation_func_list=['relu', 'relu', 'relu', 'softmax'])
    # net.init_layers(init_type='he_normal')
    #
    # j = net.CLASS_TAG
    #
    # j = net.layers[0]['weights']
    # input = Random.gaussian_distribution(15, 4, 1000)
    # input = np.asarray(input)
    #
    # plt.figure()
    # plt.hist(x=input.tolist(), bins=100)
    #
    # input = net.standardization(input)
    # input = input.tolist()
    # plt.figure()
    # plt.hist(x=input, bins=100)
    # plt.show()
    #
    # input = np.asarray([1, 2, 3, 4, 5])
    # input = np.reshape(input, newshape=(1, 5))
    # input = net.normalise(input)
    #
    # output = net.feed_foward(input)
    # net.mutate(0.5)
    # b = np.sum(output)
    #
    # net.save_model(filename="test_net")




