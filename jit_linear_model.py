# Linear Regression Own Model
# instead of copying the Google Colab model, I intend to make my own model and train it with the same data

#general
import io

# data
import numpy as np
import pandas as pd

# random numbers
import random

def mean_cuadratic_error(real_values: list, estimated_values: list):
    '''
    Calculate the Mean Cuadratic Error, given the real values and the estimated values.
    Arguments:
        - real_values: list of the real values.
        - estimated_values: list of the estimated values by the model.
        * real_values and estimated_values have to be sorted in the same order.
    '''
    mce = 0
    for i in range(len(real_values)):
        mce += (estimated_values[i] - real_values[i])**2
    mce /= (len(real_values) + 1)
    return mce

def mean_cuadratic_error_bias_gradient(real_values:list, estimated_values: list):
    '''
    Calculate the derivative of the MCE to the bias.
    Arguments:
        - real_values: list of the real values.
        - estimated_values: list of the estimated values by the model.
        * real_values and estimated_values have to be sorted in the same order.
    '''
    mce_bias_gradient = 0
    for i in range(len(real_values)):
        mce_bias_gradient += (estimated_values[i] - real_values[i]) * 2
    mce_bias_gradient /= (len(real_values) + 1)
    return mce_bias_gradient

def mean_cuadratic_error_weight_gradient(real_values:list, estimated_values: list, variables_values: list):
    '''
    Calculate the derivative of the MCE to the weight of a certain variable.
    Arguments:
        - real_values: list of the real values.
        - estimated_values: list of the estimated values by the model.
        * real_values and estimated_values have to be sorted in the same order.
    '''
    mce_weight_gradient = 0
    for i in range(len(real_values)):
        mce_weight_gradient += (estimated_values[i] - real_values[i]) * 2 * variables_values[i]
    mce_weight_gradient /= (len(real_values) + 1)
    return mce_weight_gradient

class Model():
    def __init__(self, variables_number: int, variables_labels: list, label: str):
        '''
        Initialization of Model instance. Linear model. 
        '''
        # define label of model and of variables of it
        self.label = label
        self.variables_labels = variables_labels
        self.variables_number = len(self.variables_labels)

        # initialize bias and weights
        self.bias = 0
        self.weights = []
        for i in range(self.variables_number):
            self.weights.append(1)

    def train(self, dataset: pd.DataFrame, epochs: int, batch_size: int, learning_rate: float):
        '''
        Arguments
            - dataset
            - epochs: int. Amount of epochs to do.
            - batch_size: int. Batch size to use in each iteration of training.
            - learning_rate: float. Learning rate to apply to modify gradient of loss.
        '''
        dataset_size = len(dataset.index)
        print(f"dataset size = {dataset_size}")

        epochs_trained = 0
        iteration = 0
        self.losses = []
        unused_indexes = [*range(dataset_size)]
        while (epochs_trained < epochs):
            iteration += 1
            print(f"iteration {iteration}\n")
            # select data points to use in this iteration (at random):
            #   data points are used only once per epoch
            if len(unused_indexes) < batch_size:
                epochs_trained += 1
                unused_indexes = [*range(dataset_size)]

                estimated_labels = []
                for xi in dataset_variables:
                    estimated_labels.append(self.estimate(xi))
                mce = mean_cuadratic_error(dataset[self.label], estimated_labels)
                self.losses.append(mce)
            
            data_indexes = []
            data_points = []
            data_xi = []
            
            for i in range(batch_size):
                random_number = random.randint(0, len(unused_indexes) - 1)
                index = random_number
                unused_indexes.pop(index)
                data_indexes.append(index)
                data_points.append(dataset[self.label][index])
                xi = []
                for variable in self.variables_labels:
                    xi.append(dataset[variable][index])
                data_xi.append(xi)

            # estimate values with current bias and weight:
            estimated_labels = []
            for xi in data_xi:
                estimated_labels.append(self.estimate(xi))

            # calculate MCE with current bias and weights:
            mce = mean_cuadratic_error(data_points, estimated_labels)
            print(f"Mean Cuadratic Error = {mce}\n\n")

            # calculate weights and bias gradient
            gradient_bias = mean_cuadratic_error_bias_gradient(data_points, estimated_labels)
            gradient_weights = []
            for i in range(self.variables_number):
                variable_values = []
                for j in range(batch_size):
                    variable_values.append(data_xi[j][i])
                gradient_weights.append(mean_cuadratic_error_weight_gradient(data_points, estimated_labels, variable_values))

            # calculate new weights and bias with gradients and learning_rate
            self.bias -= learning_rate * gradient_bias
            for i in range(self.variables_number):
                self.weights[i] -= learning_rate * gradient_weights[i]

    def estimate(self, variables_values):
        estimated_label = self.bias
        for i in range(self.variables_number):
            estimated_label += self.weights[i] * variables_values[i]
        return estimated_label
