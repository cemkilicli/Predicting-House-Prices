import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

test = pd.read_csv("./data/test.csv")
train = pd.read_csv("./data/train.csv")

#Separate numerical & categorical data types
set(train.dtypes.tolist())
train_num = train.select_dtypes(include = ['float64', 'int64'])
train_cat = train.select_dtypes(include = ['object'])


#Handle missing values in Training Data Set
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(train_num)
train_num = imputer.transform(train_num)

##Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()

train_cat = labelencoder_x.fit_transform(train_cat)
onehotencoder = OneHotEncoder(categorical_features =[0])

train_cat=onehotencoder.fit_transform(train_cat).toarray()

print train_cat