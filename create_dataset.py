from catboost.datasets import titanic
import pandas as pd
train, test = titanic()
train['Sex'] = train['Sex'].apply(lambda x: 0 if 'male' else 1)
test['Sex'] = test['Sex'].apply(lambda x: 0 if 'male' else 1)
train['Age'] = train['Age'].fillna(train.Age.mean()) 
test['Age'] = test['Age'].fillna(train.Age.mean())
train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']].to_csv('data_train.csv', index=False)
test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].to_csv('data_test.csv', index=False)
