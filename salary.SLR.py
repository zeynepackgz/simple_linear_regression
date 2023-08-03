# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv("Salary_dataset.csv")

tecrube = df[['YearsExperience']]
maas = df[['Salary']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(tecrube, maas, test_size= 0.2, random_state=0 ) 

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

tahmin =lr.predict(x_test)

x_train_sorted= x_train.sort_values(by='YearsExperience')
y_train_sorted=y_train.sort_values(by='Salary')

plt.plot(x_train_sorted, y_train_sorted)
plt.plot(x_test, tahmin)
plt.title('tahmin')
plt.xlabel('maas')
plt.ylabel('tecrube')
plt.legend(['veriler', 'tahmin'])








