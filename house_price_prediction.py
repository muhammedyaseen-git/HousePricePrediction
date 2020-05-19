#!/usr/bin/env python
# coding: utf-8

# #### Written by Muhammed Yaseen
# #### 111701032
# #### CSE | B.Tech | IIT-PKD

# In[1]:


import numpy as np                     #for linear algebra
import matplotlib.pyplot as plt        #helper library for plotting
import pandas as pd                    #data processing with csv files
import seaborn as sns                  #library for statistical graphics

#add plots to the jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
#from scipy import stats
from scipy.stats import norm, skew     #some math utilities


# In[2]:


#Libraries for modelling 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


# In[3]:


#load the train and test data provided into a pandas dataframe from the csv file

train = pd.read_csv('houseprice_train.csv')

test = pd.read_csv('houseprice_test.csv')


# In[4]:


train.info()        #metadata of train dataframe


# In[5]:


train.describe() #data values with some relavant statistics of the train data


# In[6]:


test.info() #metadata of test data


# In[7]:


test.describe()      ##data values with some relavant statistics of the test data


# In[8]:


print ("Size of train data : {}" .format(train.shape))

print ("Size of test data : {}" .format(test.shape))

#train data has 81 features
#test data has 80 features, excluding SalePrice
#train data has 1460 data samples
#test data has 1459 samples


# In[9]:


# a utility function to check the skewness of a feature
def check_skewness(col):
    sns.distplot(train[col] , fit=norm);   #A distplot plots a univariate distribution of observations.
    fig = plt.figure()
    (mu, sigma) = norm.fit(train[col])     #calculate mean and std.dev of the particular feature of the train data
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))


# # Cleaning the data

# In[10]:


#Save the 'Id' column for future use
train_ID = train['Id']
test_ID = test['Id']

#'Id' is dropped as it is irrelevant in the prediction process
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[11]:


print ("Size of train data after dropping Id: {}" .format(train.shape))
print ("Size of test data after dropping Id: {}" .format(test.shape))

#train has 80 features, test has 79 features excluding SalePrice


# In[12]:


'''

Dealing with outliers

Outlinear in the GrLivArea is recommended by the author of the data to remove it. 

Ref: Ames housing dataset http://jse.amstat.org/v19n3/decock.pdf

'''

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# ##### Please note the outliers towards the right.

# In[13]:


train = train[train['GrLivArea'] < 4000]
#train data is rid of the samples having GrLivArea >= 4000


# In[14]:


#check for correlation with the features
corrmat = train.corr()
#the features having corr > 0.5 and corr < -0.5 are considered seperately
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
#plot the correlation map
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ### Some points to note:
# #### 1. OverallQual, GrLivArea, TotalBsmtSF have maximum correlation with SalePrice
# #### 2. GarageArea and GarageCars have highest correlation, which can be understood from their nature. A similar pattern is observed across such data feature pairs.

# In[15]:


#print the top_corr_feature types (for our knowledge)
print(train[top_corr_features].dtypes)
#we can see that all are numerical types


# In[16]:


#A barplot to see how OverallQual depends on the SalePrice
sns.barplot(train.OverallQual,train.SalePrice)


# #### Almost a linear variance as it should be from the correlation plot

# In[17]:


#Scatter plots between 'SalePrice' and correlated variables
print(top_corr_features)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], height = 2.5)
plt.show();


# In[18]:


sns.scatterplot(train.GrLivArea,train.TotalBsmtSF)


# #### It can be observed that GrLivArea(Above grade (ground) living area square feet) is greater than 
# #### TotalBsmtSF(Total square feet of basement area)
# #### in most of the cases

# In[19]:


check_skewness('SalePrice')


# #### SalePrice or our target variable is skewed towards the left of the median as it is evident from the above plot.
# #### Regression model will not work if it is not a normal distribution as it can affect the parameter calculations, the above plot is skewed right so we need to normalize it. We use a log normalisation.

# In[20]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

check_skewness('SalePrice')


# #### We can see that the data has become fairly normal
# #### If you log transform the response variable, it is required to also log transform feature variables that are skewed. We will do it at a later stage.

# # Feature Engineering
# 
# #### We are using some data mining techniques to make the data prediction more accurate.
# #### We are concatinating test data and train data and are also remembering their respective sizes.
# #### This is necessary as all the data mining techniques done to train data should be done to the test data too.

# In[21]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values      # y_train is the target variable i.e. SalePrice and is stored seperately
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)   #SalePrice from train data is dropped for further analyses
print("all_data size is : {}".format(all_data.shape))


# ## Handling missing data

# In[22]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100  #percentage of missing data
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na}) #missing data is a dataframe having top 30 missing features
missing_data


# In[23]:


#plot missing data percentages (to make a relative comparison)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# In[24]:


all_data = all_data.drop(['PoolQC'], axis=1)

# As nearly 100% of PoolQC are missing, we can safely drop that feature


# In[25]:


all_data["Alley"] = all_data["Alley"].fillna("None")

#From the data description, NA means No alley access 


# In[26]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

#NA means None from the data description


# In[27]:


all_data["Fence"] = all_data["Fence"].fillna("None")

#NA means No Fence from the data description


# In[28]:


all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#NA means No Fireplace from the data description


# In[29]:


all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#LotFrontage(Linear feet of street connected to property) is assumed to be the median of the neighborhood properties


# In[30]:


for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    all_data[col] = all_data[col].fillna('None')
    
#NA means No Garage according to the data description


# In[31]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#Numerical features. They might be missing because garage may not be present at all. We are assuming them to be zero


# In[32]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
#Similar explanation as the above one


# In[33]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

#Categorical features, similar explanation as above


# In[34]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#MasVnrArea: Masonry veneer area in square feet
#Similar explanation


# In[35]:


all_data['Functional'].value_counts()


# In[36]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")

#Home functionality: Replaced with Typ, the most occured value


# In[37]:


mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_col:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
#Similar to functionality, these features are replaced with their mode values


# In[38]:


all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# In[39]:


all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# In[40]:


#A sanity check for missing data to confirm 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

missing_data


# #### We have accounted for all the missing features as the above dataframe is empty.

# ### Removing redundant data (these were identified from the skew)

# In[41]:


all_data['Utilities'].value_counts()


# In[42]:


all_data['MiscVal'].value_counts()


# In[43]:


all_data = all_data.drop(['Utilities'], axis=1)
all_data = all_data.drop(['MiscVal'], axis=1)

#Except for one, all the other samples have the same value.
#Irrelevant for analysis, dropping it


# ### Accounting for categorical features

# #### The data in the following group, all have categorical variables disguised in number format.
# #### We have to change them to the string type

# In[44]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# #### LabelEncoder encode labels with a value between 0 and n_classes-1 where n is the number of distinct labels. 
# #### If a label repeats it assigns the same value to as assigned earlier.
# 

# In[45]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# In[46]:


# Adding total sqfootage feature , Usually houses are categorised by area
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[47]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features using skew utility
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)
#top 15


# In[48]:


#applying log tranformation where skewness > 0.75 and skewness < -0.75
skewness = skewness[abs(skewness) > 0]
skewed_features = skewness.index
for feat in skewed_features:
    all_data[feat] = np.log1p(all_data[feat])


# In[49]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness[abs(skewness) > 0.75]
#print(skewness[skewness[‘’]==’SalePrice’].index.values)


# #### A noticeable improvement is seen with the skewed data

# In[50]:


all_data = pd.get_dummies(all_data)
all_data.shape


# #### Earlier, we had concatenated train and test data for feature engineering purposes
# #### We have to split it back to the original form for modelling purposes

# In[51]:


train = all_data[:ntrain]
test = all_data[ntrain:]
train.shape


# # Modelling
# 
# ### As the problem involves predicting a variable wrt. other variables, we will use mutivariate linear regression models.
# ### Also, to tackle the cases of overfitting, we arrive at Lasso and ridge regressions to choose from.
# ### The methods are chosen only from ones those were covered in the class.
# 
# #### best_score_: Mean cross-validated score of the best_estimator. Score here means the R2 score.

# In[52]:


linear_reg = LinearRegression(normalize = True)
parameters = [0.0001, 0.001, 0.003, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.5, 1, 5, 10]
param_grid = {'alpha' : parameters}
linear_cv = GridSearchCV(linear_reg, param_grid = {}, cv = 3)

linear_cv.fit(train, y_train)
# Print the tuned parameters and score 
print("Result of OLS Regression:\n")
#print("Tuned Logistic Regression Parameters: {}".format(linear_cv.best_params_))  
print("Best score is {}".format(linear_cv.best_score_)) 


# In[53]:


pd.DataFrame(linear_cv.cv_results_)


# #### OLS is discarded due to the negative score.

# #### Regression is carried out with GridSearchCV utility function which performs a cross validation and selects the hyperparameter 'alpha'. Here, the cross validation is 3-fold (project guideline instructs to choose a roughly 70-30 train-test split).
# 
# ### Lasso Regression

# In[54]:


lasso = Lasso()
parameters = [0.0001, 0.001, 0.003, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.5, 1, 5, 10]
param_grid = {'alpha' : parameters}
# Instantiating the GridSearchCV object 
lasso_cv = GridSearchCV(lasso, param_grid, cv = 3) 
  
lasso_cv.fit(train, y_train)
  
# Print the tuned parameters and score 
print("Result of Lasso Regression:\n")
print("Tuned Lasso Regression Parameters: {}".format(lasso_cv.best_params_))  
print("Best score is {}".format(lasso_cv.best_score_)) 


# #### Lasso feature extraction

# In[55]:


null_coeffs = pd.Series(lasso_cv.best_estimator_.coef_, index=train.columns)
res = null_coeffs.to_list()
zero_count = 0
for val in res:
    if val == 0:
        zero_count += 1
print("Lasso eliminated ",zero_count," variables out of ",len(res)," variables")


# ### Ridge Regression

# In[56]:


ridge = Ridge()
parameters = [0.0001, 0.001, 0.003, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
param_grid = {'alpha' : parameters}
# Instantiating the GridSearchCV object 
ridge_cv = GridSearchCV(ridge, param_grid, cv = 3) 
  
ridge_cv.fit(train, y_train)
  
# Print the tuned parameters and score 
print("Tuned Ridge Regression Parameters: {}".format(ridge_cv.best_params_))  
print("Best score is {}".format(ridge_cv.best_score_)) 


# In[57]:


#lasso performs better
lasso_cv.fit(train,y_train)
ridge_cv.fit(train,y_train)


# #### As LASSO and RIDGE models perform more or less similar, we shall take a weighted average of them to be the final predictor. Since, RIDGE has a marginally high score, we are giving a higher priority for RIDGE model.
# 
# #### Finally, SalePrice is converted back to its original form from the logarithmic transformed form

# In[58]:


final_model = (0.45*np.expm1(lasso_cv.predict(test)) + 0.55*np.expm1(ridge_cv.predict(test)) )


# In[59]:


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = final_model
sub.to_csv('submission_project_final.csv',index=False)


# In[60]:


#### The entry got top 22% in global Kaggle Leaderboard with a rank 1084 and a score 0.12365.
#### https://kaggle.com/c/house-prices-advanced-regression-techniques
#### Improvements are possible with advanced regression techniques

