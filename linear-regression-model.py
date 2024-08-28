# Linear Regression Own Model
# instead of copying the Google Colab model, I intend to make my own model and train it with the same data

#general
import io

# data
import numpy as np
import pandas as pd
# from jit-linear-model import *
import jit_linear_model

import matplotlib.pyplot as plt

# import dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# update dataframe to use certain columns
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

linear_model_1var = jit_linear_model.Model(1, ['TRIP_MILES'], 'FARE')
linear_model_1var.train(dataset = training_df, epochs = 3, batch_size = 400, learning_rate=0.0001)
print(linear_model_1var.bias)
print(linear_model_1var.weights)
x = range(len(linear_model_1var.losses))
plt.plot(x, linear_model_1var.losses)
plt.show()

# ADD PLOT OF MODEL VS REAL DATA
