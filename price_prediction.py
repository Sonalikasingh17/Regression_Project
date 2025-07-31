import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read.csv('laptop_data.csv')

df.head(10)

df.shape

df.info()

df.duplicated().sum()

df.isnull().sum()

df.drop(coloumns = ['unnamed : 0'], inplace = 'true')
