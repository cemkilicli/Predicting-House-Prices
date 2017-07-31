import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

test = pd.read_csv("./data/test.csv")
train = pd.read_csv("./data/train.csv")

# Check how much data there is
#print train.info()
print train.shape
#print train.head()

set(train.dtypes.tolist())
dfnum = train.select_dtypes(include = ['float64', 'int64'])
dfcat = train.select_dtypes(include = ['object'])
print "Numerical data", dfnum.info()
print "Categorical data",dfcat.info()


# Check how much data there is
print test.info()
print test.shape
print test.head()

labels = list(train)


#Checking for missing data
NAN_s = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
print NAN_s[NAN_s.sum(axis=1) > 0]

#Select numerical data types
set(train.dtypes.tolist())
dfnum = train.select_dtypes(include = ['float64', 'int64'])
print dfnum.info()


#Plot the SalePrice to understand the skewness of data
sns.distplot(train[['SalePrice']], color = 'g', bins = 100)
plt.show()

#Find correlation
dfnum_corr = dfnum.corr()['SalePrice'][:-1]
golden_feature_list = dfnum_corr[abs(dfnum_corr) > 0.5].sort_values(ascending = False)
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))

#Plot all the numerical variables
cm1 = dfnum.hist(figsize = (16,20), bins = 50, xlabelsize =8, ylabelsize = 8)
plt.show()


dfnum.plot.scatter(x = '1stFlrSF', y = 'SalePrice')
dfnum.plot.scatter(x = 'GrLivArea', y = 'SalePrice')
sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = dfnum)
dfnum.plot.scatter(x = 'GarageArea', y = 'SalePrice')
dfnum.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')
dfnum.plot.scatter(x = '1stFlrSF', y = 'SalePrice')
sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = dfnum)
sns.boxplot(x = 'YearRemodAdd', y = 'SalePrice', data = dfnum)
sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = dfnum)
sns.boxplot(x = 'FullBath', y = 'SalePrice', data = dfnum)
sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = dfnum)
plt.show()


#Delete the outliers
dfnum = dfnum.drop(dfnum[dfnum['Id'] == 1299].index)
dfnum = dfnum.drop(dfnum[dfnum['Id'] == 524].index)



#Create Heatmap of Correlated variables
dfnum_corr1 = dfnum.corr()
cols = dfnum_corr1.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(dfnum[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


