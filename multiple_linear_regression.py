import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('.//50_Startups.csv')
#print(df.head())

X = df.iloc[:,:-1].values
y = df.iloc[:,4].values

#print(X)
#print(y)

# lets as consider a categorical data

group = df.groupby('State')

"""for name in group:

    print(name)
"""
#this block is for categorical data because in our dataset state columns have string value to change in numerical data in form of 0 and1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()



#now  data in train and test where test_size is 20%
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#now fit a in linear model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)



# now prediction a X_test
print(reg.predict(X_test))


