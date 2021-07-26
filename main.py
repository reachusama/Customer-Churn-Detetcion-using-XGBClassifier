import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ForTestDataPrep import DfPrepPipeline, best_model, get_auc_scores

# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('Dataset/Churn_Modelling.csv', delimiter=',')

# 1000 * 14
# print(df.shape)

# Any missing values
# print(df.isnull().sum())
# print(df.nunique())

# Droping data related to customer or data tracking
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
# print(df.head())
# print(df.dtypes)

# Proportion of customer churned and retained
# sizes = [df.Exited[df['Exited'] == 1].count(), df.Exited[df['Exited'] == 0].count()]
# print(sizes)

# labels = 'Exited', 'Retained'
# explode = (0, 0.1)
# fig1, ax1 = plt.subplots(figsize=(10, 8))
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# ax1.axis('equal')
# plt.title("Proportion of customer churned and retained", size=20)
# plt.show()


# Relation of categorical variables with the customer churn and retaining
# fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
# sns.countplot(x='Geography', hue='Exited', data=df, ax=axarr[0][0])
# sns.countplot(x='Gender', hue='Exited', data=df, ax=axarr[0][1])
# sns.countplot(x='HasCrCard', hue='Exited', data=df, ax=axarr[1][0])
# sns.countplot(x='IsActiveMember', hue='Exited', data=df, ax=axarr[1][1])


# Relations based on the continuous data attributes
# fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
# sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = df, ax=axarr[0][0])
# sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = df , ax=axarr[0][1])
# sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][0])
# sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = df, ax=axarr[1][1])
# sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][0])
# sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = df, ax=axarr[2][1])
# plt.show()

df_train = df.sample(frac=0.8, random_state=200)
df_test = df.drop(df_train.index)
# print(len(df_train))
# print(len(df_test))

# the ratio of the bank balance and the estimated salary
df_train['BalanceSalaryRatio'] = df_train.Balance / df_train.EstimatedSalary
# sns.boxplot(y='BalanceSalaryRatio', x='Exited', hue='Exited', data=df_train)
# plt.ylim(-1, 5)
# plt.show()

# Given that tenure is a 'function' of age, introducing a variable aiming to standardize tenure over age:
df_train['TenureByAge'] = df_train.Tenure / (df_train.Age)
# sns.boxplot(y='TenureByAge', x='Exited', hue='Exited', data=df_train)
# plt.ylim(-1, 1)
# plt.show()

# capture credit score given age to take into account credit behaviour visavis adult life
df_train['CreditScoreGivenAge'] = df_train.CreditScore / (df_train.Age)
# sns.boxplot(y='CreditScoreGivenAge', x='Exited', hue='Exited', data=df_train)
# plt.ylim(-1, 40)
# plt.show()

continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                   'TenureByAge', 'CreditScoreGivenAge']
cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography', 'Gender']
df_train = df_train[['Exited'] + continuous_vars + cat_vars]
# print(df_train.head())

# for the one hot variables, we change 0 to -1 so that the models can capture a negative relation
df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
print(df_train.head())

# One hot encode the categorical variables
lst = ['Geography', 'Gender']
remove = list()
for i in lst:
    if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
        for j in df_train[i].unique():
            df_train[i + '_' + j] = np.where(df_train[i] == j, 1, -1)
        remove.append(i)

df_train = df_train.drop(remove, axis=1)
# print(df_train.head())

# minMax scaling the continuous variables
minVec = df_train[continuous_vars].min().copy()
maxVec = df_train[continuous_vars].max().copy()
df_train[continuous_vars] = (df_train[continuous_vars] - minVec) / (maxVec - minVec)
# print(df_train.head())


# Fit Extreme Gradient Boost Classifier
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0.01,
                    learning_rate=0.1, max_delta_step=0, max_depth=7,
                    min_child_weight=5, missing=1, n_estimators=20, n_jobs=1, nthread=None,
                    objective='binary:logistic', random_state=0, reg_alpha=0,
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)

XGB.fit(df_train.loc[:, df_train.columns != 'Exited'], df_train.Exited)

df_test = DfPrepPipeline(df_test, df_train.columns, minVec, maxVec)
df_test = df_test.mask(np.isinf(df_test))
df_test = df_test.dropna()
# print(df_test.isnull().sum())

df_pred = XGB.predict(df_test.loc[:, df_test.columns != 'Exited'])
print(classification_report(df_test.Exited, df_pred))
