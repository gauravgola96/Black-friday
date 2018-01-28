

import pandas as pd
import numpy as np

train = pd.read_csv("C:\\Users\\Gaurav_Gola\\Desktop\\project\\black friday\\train.csv")

test = pd.read_csv("C:\\Users\\Gaurav_Gola\\Desktop\\project\\black friday\\test.csv")

train.shape

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train,test],ignore_index=True)


data

train.shape

data.dtypes

train

test.isnull().sum()

train.isnull().sum()

len(train.Product_Category_1.unique())

# total categories can 20 ..
#products will belong to these category

train.Product_Category_1.unique()

len(train.Age.unique())

len(train.Stay_In_Current_City_Years.unique())


data.Product_Category_2 = data.Product_Category_2.fillna(value=0)

data.Product_Category_3 = data.Product_Category_3.fillna(value=0)

data.Occupation.unique()

data # filling with zero introduces a new category 
# this means prod_cat with zero value are not included for that product

type(data)

import seaborn as sn
import matplotlib.pyplot as plt
%pylab inline
%matplotlib inline

fig,ax = plt.subplots()

#?sn.boxplot

sn.boxplot(data=data[["Purchase","Product_Category_1"]],x="Product_Category_1",y="Purchase")
#no effect

sn.barplot(data=data[["Purchase","Stay_In_Current_City_Years"]],x="Stay_In_Current_City_Years",y="Purchase")
#equal in very bar



def fit_transform_ohe(df,col_name):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le_lable = le.fit_transform(df[col_name])
    df[col_name+'_label']=le_lable
    ## one hot encoding 
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    return(le,ohe,features_df)


#data = data.applymap("str")

data.Product_Category_1 =data.Product_Category_1.astype("category")
data.Product_Category_2 = data.Product_Category_2.astype("category")
data.Product_Category_3 = data.Product_Category_3.astype("category")

data.Age = data.Age.astype("category")
data.Gender = data.Gender.astype("category")
data.City_Category = data.City_Category.astype("category")
data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.astype("category")
data.Marital_Status = data.Marital_Status.astype("category")
data.Occupation = data.Occupation.astype("category")


data.dtypes

cat_atr = [x for x in data.dtypes.index if data.dtypes[x]=="object"]
cat_atr

data_1 = data.drop(["Product_ID","User_ID","Purchase","source",],axis=1)

cat_var= data_1.columns
cat_var

cat_varr = cat_var.drop(["Marital_Status","Occupation"])
cat_varr

encoded_attr_list = []
for col in cat_var:
    return_obj = fit_transform_ohe(df=data,col_name=col)
    encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})

numeric_feature_col = [x for x in data.dtypes.index if data.dtypes[x]=="int64" or  data.dtypes[x]=="float64"]

numeric_feature_col

feature_df_list = [data[numeric_feature_col]]
feature_df_list.extend([enc['feature_df'] \
for enc in encoded_attr_list \
if enc['col_name'] in cat_var])
train_df_new = pd.concat(feature_df_list, axis=1)
print("Shape::{}".format(train_df_new.shape))

target = data.Purchase

target_array =  np.array(target)

train_df_new_without_target = train_df_new.drop(["Purchase"],axis=1)

train_df_new_array = np.array(train_df_new_without_target)

target.isnull().sum()

train_df_new_array[0:233599]

#Divide it back :-
#Divide into test and train:
#train = data.loc[data['source']=="train"]
#test = data.loc[data['source']=="test"]


train_final = train_df_new_array[data['source']=="train"]
train_final

train_target = target[data['source']=="train"]

test_final = train_df_new_array[data['source']=="test"]

test_target = target[data["source"]=="test"]

import sklearn
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train_final,train_target,test_size=0.33,random_state=42)

?train_test_split



import sklearn
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

from sklearn.cross_validation import cross_val_score,cross_val_predict

train_predict = cross_val_predict(estimator=lr,X=X_train,y=Y_train,cv=5)

train_predict

train_score_predict = cross_val_score(estimator=lr,X=X_train,y=Y_train,cv=5)



lr.fit(X=X_train,y=Y_train)






train_predict=lr.predict(X_test)
train_predict



from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import mean_squared_error
from math import sqrt



rms = sqrt(mean_squared_error(Y_test, train_predict))

rms

mse = mse(y_true=Y_test,y_pred=train_predict,sample_weight=None, multioutput='uniform_average')

sqrt(mse)

from sklearn.linear_model import Ridge
Rd = Ridge(alpha=0.05,normalize=True)

Rd.fit(X=X_train,y=Y_train)

train_predict_Rd=Rd.predict(X_test)
train_predict_Rd

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(Y_test, train_predict_Rd))
rms # RMSE increased

from sklearn.neighbors import KNeighborsRegressor

Knn = KNeighborsRegressor()

Knn.fit(X=X_train,y=Y_train)

train_predict_KNN=lr.predict(X_test)
train_predict_KNN

train_predict_KNN.shape



#UDF for RMSE 
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse(predictions=train_predict_KNN,targets=Y_test)
#same as linear regression 

# NO change in RMSE 
# So No use dummy variable
# Do the lable encoder

#Convert all the categorical to numerical using Label encoder 
#re-run the code upto missin values imputation

target = data.Purchase

target = np.array(target)

data.drop(["Purchase"],axis=1,inplace=True)
data

from sklearn.preprocessing import LabelEncoder
Lb = LabelEncoder()


data_copy = data.copy() #backup
data.dtypes

#Convert all the columns to string 
data = data.applymap(str)

data.dtypes



input = np.array(data)


input # rows * 13 [ columns]
input.shape[1]

for i in range(input.shape[1]):
    lbl = sklearn.preprocessing.LabelEncoder()
    lbl.fit(list(input[:,i]))
    input[:, i] = lbl.transform(input[:, i])

input.astype(int)

target

train_final = input[data['source']=="train"]
train_target = target[data['source']=="train"]
test_final = input[data['source']=="test"]
test_target = target[data["source"]=="test"]

test_target

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train_final,train_target,test_size=0.33,random_state=42)





from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()

RF.fit(train_final,train_target)

train_predict_RF = RF.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(Y_test, train_predict_RF))
rms 

submission.to_csv("../submission/submit_13.csv", index=False)