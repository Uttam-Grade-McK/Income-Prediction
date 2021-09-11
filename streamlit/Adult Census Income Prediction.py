import pandas as pd
import numpy as np
import plotly.express as px
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("adult.csv")
print(df.shape)
print(df.describe())
print(df.info())

salary_map = {" <=50K" :1, " >50K": 0}
df['salary'] = df['salary'].map(salary_map).astype(int)

sex_map = {" Male" :1, " Female": 0}
df['sex'] = df['sex'].map(sex_map).astype(int)


df['race'].unique()

def plot_correlation(df, size=15):
    corr= df.corr()
    fig, ax = plt.subplots(figsize=(size,size))
    
    ax.matshow(corr)        
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.show()
    
def plot_corr(df, size= 14):
    plt.subplots(figsize=(15,15))
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    
plot_corr(df)

plot_correlation(df)
df['country'].unique()
df[['country','salary']].groupby(['country']).count()

df['country'].value_counts()
df['country'][0]

def mark(x):
    if x in " United-States":
        return 'United-States'
    else:
        return 'Non-US'
    
df['country'] = df['country'].apply(mark)
print(df[['country','salary']].groupby(['country']).mean())
values=df['country'].value_counts()
labels=df['country'].value_counts().index
df['country'].unique()
df['country'] = df['country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)

df.dropna(how='any',inplace=True)
fig = px.pie(df, values=values, names=labels,title='Pie chart')
fig.show()
df['country'] = df['country'].map({'United-States':1,'Non-US':0}).astype(int)
x= df['hours-per-week']
plt.hist(x,bins=None,density=True,histtype='bar')
plt.show()

df[['relationship','salary']].groupby(['relationship']).mean()
df[['marital-status','salary']].groupby(['marital-status']).mean()
df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
df[['marital-status','salary']].groupby(['marital-status']).mean()
values=df['marital-status'].value_counts()
labels=df['marital-status'].value_counts().index
fig = px.pie(df, values=values, names=labels,title='Pie chart')
fig.show()

df[['marital-status','relationship','salary']].groupby(['marital-status','relationship']).mean()
df[['marital-status','relationship','salary']].groupby(['relationship','marital-status']).mean()
df['marital-status'] = df['marital-status'].map({'Couple':0,'Single':1})
rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}

df['relationship'] = df['relationship'].map(rel_map)
df[['race','salary']].groupby('race').mean()
race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}


df['race']= df['race'].map(race_map)
df[['occupation','salary']].groupby(['occupation']).mean()
df[['workclass','salary']].groupby(['workclass']).mean()

def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov': return 'govt'
    elif x['workclass'] == ' Private':return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc': return 'self_employed'
    else: return 'without_pay'
    
    
df['employment_type']=df.apply(f, axis=1)   
df[['employment_type','salary']].groupby(['employment_type']).mean()
employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}

df['employment_type'] = df['employment_type'].map(employment_map)
df[['education','salary']].groupby(['education']).mean()
df.drop(labels=['workclass','education','occupation'],axis=1,inplace=True)
x= df['education-num']
plt.hist(x,bins=None,density=True,histtype='bar')
plt.show()

x=df['capital-gain']
plt.hist(x,bins=None)
plt.show()

df.loc[(df['capital-gain'] > 0),'capital-gain'] = 1
df.loc[(df['capital-gain'] == 0 ,'capital-gain')]= 0

values=df['capital-gain'].value_counts()
labels=df['capital-gain'].value_counts().index
fig = px.pie(df, values=values, names=labels,title='Pie chart')
fig.show()



from sklearn.model_selection import train_test_split

X= df.drop(['salary'],axis=1)
y=df['salary']

split_size=0.3

#Creation of Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_size,random_state=22)

#Creation of Train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)

print ("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print ("Validation dataset: {0}{1}".format(X_val.shape, y_val.shape))
print ("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = []
names = ['LR','Random Forest','Neural Network','GaussianNB','DecisionTreeClassifier','SVM',]

models.append((LogisticRegression()))
models.append((RandomForestClassifier(n_estimators=100)))
models.append((MLPClassifier()))
models.append((GaussianNB()))
models.append((DecisionTreeClassifier()))
models.append((SVC()))

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
kfold = model_selection.KFold(n_splits=5)

for i in range(0,len(models)):    
    cv_result = model_selection.cross_val_score(models[i],X_train,y_train,cv=kfold,scoring='accuracy')
    score=models[i].fit(X_train,y_train)
    prediction = models[i].predict(X_val)
    acc_score = accuracy_score(y_val,prediction)     
    print ('-'*40)
    print ('{0}: {1}'.format(names[i],acc_score))

randomForest = RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train,y_train)
prediction = randomForest.predict(X_test)
import pickle
pickle_out = open('Income_Classifier.pkl', 'wb')
pickle.dump(randomForest, pickle_out)
pickle_out.close()
randomForest.predict([[20, 586, 8, 1, 2, 3, 1, 1, 0, 70, 1, 1]])
print(df.head(5))


























