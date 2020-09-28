# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:56:45 2020

@author: manji
"""

##Multi collinearity in Linear Regression

import pandas as pd
import statsmodels.api as sm

df_adv = pd.read_csv('C:/Users/manji/Downloads/Multicollinearity-master/data/Advertising.csv')
X = df_adv.iloc[:,:-1]
y = df_adv.iloc[:,5]

##fit a OLs(ordinary Linear Squares) with intercept on TV and radio
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

model.summary()

import matplotlib.pyplot as plt
X.iloc[:,1:].corr()

##Salary Data to find multi collinearity 

df_salary = pd.read_csv('C:/Users/manji/Downloads/Multicollinearity-master/data/Salary_Data.csv')
X1 = df_salary.iloc[:,:-1]
y1 = df_salary.iloc[:,2]

X1 = sm.add_constant(X1)
model = sm.OLS(y1, X1).fit()

model.summary()

X1.iloc[:,1:].corr()
