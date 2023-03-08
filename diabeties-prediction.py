import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection & Analysis
# We have PIMA Diabetes Dataset with us

diabetes_dataset = pd.read_csv('/Users/raghavsharma/Documents/Programming/Python/diabetes.csv')

print("\nprinting the first 5 rows of the dataset\n")
print(diabetes_dataset.head())

# number of rows and Columns in this dataset
print("\nRows & Colums = "+str(diabetes_dataset.shape))

print("\n\ngetting the statistical measures of the data\n\n")
print(diabetes_dataset.describe())