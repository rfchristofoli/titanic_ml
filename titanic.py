# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:35:58 2020

@author: Rafael Christofoli de Barros
"""
# importando bibliotecas
#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier

#importando os dados de treino e de teste
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#plotando o heatmap
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(), cmap="coolwarm", fmt='.2f', linewidths=0.1,
            linecolor='white', vmax=1.0, square=True, annot=True)

#plotando os gráficos para Survived vs. Sex, Pclass e Embarked
fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(12,4))
sns.barplot(x='Sex', y='Survived', data=train, ax=axis1)
sns.barplot(x='Pclass', y='Survived', data=train, ax=axis2)
sns.barplot(x='Embarked', y='Survived', data=train, ax=axis3)

#salvando PasssengerId do test para submissão posterior
test_passengerId = test['PassengerId']

#removendo os campos que nao achei significantes
train_adjusted = train.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis=1)
test_adjusted = test.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis=1)

#verificando os valores nulos 
train_adjusted.isnull().sum().sort_values(ascending=False).head(10)
test_adjusted.isnull().sum().sort_values(ascending=False).head(10)

#ajustando nulos Age pela mediana do treino
train_adjusted['Age'].fillna(train_adjusted['Age'].mean(), inplace=True)
test_adjusted['Age'].fillna(train_adjusted['Age'].mean(), inplace=True)

#transformando variáveis categóricas em númericos utilizando OneHotEncoding
dummies_train = []
cols = ['Pclass','Sex','Embarked']

for col in cols: dummies_train.append(pd.get_dummies(train_adjusted[col]))
    
train_dummies = pd.concat(dummies_train, axis=1)   
train_adjusted = pd.concat((train_adjusted,train_dummies), axis=1) 
train_adjusted = train_adjusted.drop(cols, axis=1)

dummies_test = []
cols = ['Pclass','Sex','Embarked']

for col in cols: dummies_test.append(pd.get_dummies(test_adjusted[col]))
    
test_dummies = pd.concat(dummies_test, axis=1)   
test_adjusted = pd.concat((test_adjusted, test_dummies), axis=1) 
test_adjusted = test_adjusted.drop(cols, axis=1)

#plotando o heatmap do train_encoded
plt.figure(figsize=(10,10))
sns.heatmap(train_adjusted.corr(), cmap="coolwarm", fmt='.2f', linewidths=0.1,
            linecolor='white', vmax=1.0, square=True, annot=True)

#pré-processamento utilizando Standartization    
#scaler = preprocessing.StandardScaler().fit(train_adjusted[['Age']])
#train_adjusted[['Age']] = scaler.transform(train_adjusted[['Age']])
#test_adjusted[['Age']] = scaler.transform(test_adjusted[['Age']])

#separando os dados para executar treino e previsão
X = train_adjusted.drop('Survived', axis=1)
Y = train_adjusted['Survived']

#preparando GradientBoostingClassifier e realizando o treinamento
#gbk = GradientBoostingClassifier()
#gbk.fit(X, Y)

#executando previsão
#train_pred = gbk.predict(X) 

#calculando e exibindo precisão
#print(round(accuracy_score(train_pred, Y) * 100, 2))

#verificando se eu morri
#rafael = np.array([2,33,1,0,0,1,0,1,0]).reshape((1,-1))
#print("Rafael: ", gbk.predict(rafael)[0])

#verificando se a patroa morreu
#bruna = np.array([2,26,1,0,1,0,0,1,0]).reshape((1,-1)) 
#print("Bruna: ", gbk.predict(bruna)[0])

#preparando RandomForestClassifier e realizando o treinamento
clf = RandomForestClassifier()
clf.fit(X, Y)

#executando previsão
train_pred = clf.predict(X)

#calculando e exibindo precisão e as features importantes
print(round(accuracy_score(train_pred, Y) * 100, 2))
clf.feature_importances_

#executando a previsão com os dados do test  
#test_pred = gbk.predict(test_adjusted)
test_pred = clf.predict(test_adjusted)

#gerando o CSV para submeter no Kaggle
submission = pd.DataFrame({
        "PassengerId" : test_passengerId,
        "Survived" : test_pred
        })

submission.to_csv('./submission_Rafael_Christofoli.csv', index=False)
