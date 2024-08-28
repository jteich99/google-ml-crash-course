# import required packages from the exercise GoogleColab
#general
import io

# data
import numpy as np
import pandas as pd

# machine learning
# import keras

# data visualization
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# import dataset
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# update dataframe to use certain columns
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

# view dataset statistics
# use DataFrame.describe to view descriptive statistics about the dataset
print(training_df.describe(include = 'all'))

# use some of pandas functions to answer common questions
# What is the maximum fare?
print("What is the maximum fare?")
print(training_df['FARE'].max())

# What is the mean distance across all trips?
print("What is the mean distance across all trips?")
print(training_df['TRIP_MILES'].mean())

# How many cab companies are in the dataset?
print("How many cab companies are in the dataset?")
print(training_df['COMPANY'].nunique())

# What is the most frequent payment type?
print("What is the most frequent payment type?")
print(training_df['PAYMENT_TYPE'].value_counts().idxmax())

# Are any features missing data?
print("Are any features missing data?")
print(f"Missing values = {training_df.isnull().sum().sum()}")

# generate a correlation matrix
correlation_matrix = training_df.corr(numeric_only = True)
print(correlation_matrix)

# Which feature correlates most strongly to the label FARE? --> TRIP_MILES
# Which feature correlates least strongly to the label FARE? --> TIP_RATE 
# There is high correlation in between TRIP_MILES, TRIP_SECONDS and FARE, and low correlation with TIP_RATE


# generate a pair plot
# generate a pairplot using seaborn. This is a plot that I could do in matplotlib, but it would take lots of code. Here it is implemented easily to take de dataframe and visualize its data
sns.pairplot(training_df, x_vars = ["FARE",  "TRIP_MILES", "TRIP_SECONDS", "TIP_RATE"], y_vars =["FARE",  "TRIP_MILES", "TRIP_SECONDS", "TIP_RATE"])
plt.show()
# sns.show()
