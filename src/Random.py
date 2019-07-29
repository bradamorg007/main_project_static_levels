# alt 3 = # comment symbol
import math
import numpy as np
import matplotlib.pyplot as plt

class Random:

    @staticmethod
    def dim_check(x):
        if len(x) <= 1:
            raise ValueError("Error: numpy 0 dimension detected, please use np.expand_dims on the zero axis to"+
                             "to make the np array shape a tuple e.g (1, 5) ")

    def type_check(x):
        if not isinstance(x, np.ndarray):
            raise ValueError("Illegal Argument Exception: inputs must be a numpy array")



    @staticmethod
    def integer(input, min, max):

        Random.type_check(input)
        Random.dim_check(input.shape)
        rows = input.shape[0]
        cols = input.shape[1]

        for i in range(rows):
            for j in range(cols):
                input[i][j] = np.random.randint(low=min, high=max)

        return input


    @staticmethod
    def uniform_random(input):
        # uniform distribution of numbers between -1 and +1
        Random.type_check(input)
        Random.dim_check(input.shape)
        rows = input.shape[0]
        cols = input.shape[1]

        for i in range(rows):
            for j in range(cols):
                input[i][j] = np.random.rand() * 2 - 1

        return input


    @staticmethod
    def normal(input):
        # for tanh activation function
        Random.type_check(input)
        Random.dim_check(input.shape)
        rows = input.shape[0]
        cols = input.shape[1]

        input = Random.gaussian_distribution(mean=0, sigma=1, samples=input.size)
        input = np.reshape(input, newshape=(rows, cols))

        return input

    @staticmethod
    def he_normal(input):
        # for relu activation function
        Random.type_check(input)
        Random.dim_check(input.shape)
        rows = input.shape[0]
        cols = input.shape[1]

        input = Random.truncated_gaussian_distribution(mean=0, sigma=1, samples=input.size, cuttoff=2)
        input = np.multiply(input, math.sqrt(2 / cols))
        input = np.reshape(input, newshape=(rows, cols))

        return input


    @staticmethod
    def xavier_normal(input):
        # for tanh activation function
        Random.type_check(input)
        Random.dim_check(input.shape)
        rows = input.shape[0]
        cols = input.shape[1]

        input = Random.truncated_gaussian_distribution(mean=0, sigma=1, samples=input.size, cuttoff=2)
        input = np.multiply(input, math.sqrt(1 / cols))
        input = np.reshape(input, newshape=(rows, cols))

        return input


    @staticmethod
    def truncated_normal(input):
        Random.type_check(input)
        Random.dim_check(input.shape)
        rows = input.shape[0]
        cols = input.shape[1]

        input = Random.truncated_gaussian_distribution(mean=0, sigma=1, samples=input.size, cuttoff=2)

        return np.reshape(input, newshape=(rows, cols))



    @staticmethod
    def truncated_gaussian_distribution(mean, sigma, samples, cuttoff):
        if not isinstance(samples, int):
            raise ValueError(
                "Illegal Argument Exception: Number of samples must be an int")

        if samples == 0:
            raise ValueError("Illegal Argument Exception: sample number must be greater than 0")

        output = []
        for i in range(samples):
            done = False

            while done == False:
                draw = Random.gaussian_distribution(mean, sigma, 1)

                if draw <= (mean + cuttoff) and draw >= (mean - cuttoff):

                    if samples == 1:
                        return draw

                    output.append(draw)

                    done = True

        return output

    @staticmethod
    def gaussian_distribution(mean, sigma, samples):

        if not isinstance(samples, int):
            raise ValueError(
                "Illegal Argument Exception: Number of samples must be an int")

        # loop over number of samples needed

        two_pi = math.pi * 2
        output = []


        lens = samples/2
        splice_flag = False
        if lens % 1 != 0:
            lens = math.floor(lens) + 1
            splice_flag = True

        lens = int(lens)
        for i in range(lens):

            u1 = np.random.rand()
            u2 = np.random.rand()

            z = 0
            if u1 != 0:
                z = math.sqrt(-2 * math.log(u1)) * math.cos(two_pi * u2)
                output.append((z * sigma) + mean)

            else:
                output.append(0)

            if samples == 1:
                return output[0]

            if u2 != 0:
                z = math.sqrt(-2 * math.log(u1)) * math.sin(two_pi * u2)
                output.append((z * sigma) + mean)

            else:
                output.append(0)

        if splice_flag:
            output.pop()

        return output


if __name__ == "__main__":

    dist = np.zeros(shape=(100, 100))
    dist = Random.integer(dist, 0, 100)

    dist = dist.flatten(order="C")
    plt.hist(dist.tolist(), bins=100)
    plt.show()