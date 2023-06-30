# Salary_data_Preprocessing
Preprocessing the salary data before feeding it to Machine Learning algorithm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("/content/Placement_Dataset.csv")
dataset.head()
dataset.isnull().sum()

#missing values can be replaced using the following techniques
#mean values
#median
#mode
#constant value
fig, ax = plt.subplots(figsize = (8,8))
sns.distplot(dataset.salary)

dataset['salary'].fillna(dataset['salary'].median(),inplace=True)
dataset.isnull().sum()

salary_dataset = pd.read_csv('/content/Placement_Dataset.csv')
salary_dataset.shape
salary_dataset.isnull().sum()

#dropping the missing values
salary_dataset = salary_dataset.dropna(how = "any")
salary_dataset.isnull().sum()
salary_dataset.shape
