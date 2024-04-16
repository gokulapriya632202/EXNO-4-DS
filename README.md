# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns

df=pd.read_csv("/content/bmi.csv")
df1=pd.read_csv("/content/bmi.csv")
df2=pd.read_csv("/content/bmi.csv")

df.head()
```
![Screenshot 2024-04-16 155945](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/a502c491-1288-4c08-beb8-87cd12ee01d0)

```
df.dropna()
```
![Screenshot 2024-04-16 160005](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/9af94418-b007-454b-85f9-418e6cf097b9)

```
max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2024-04-16 160012](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/6d0d2bc0-239a-4954-92ea-5f85e4fbcea0)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![Screenshot 2024-04-16 160023](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/6a3e22b6-1122-4c98-a24d-4c19f0e0bfce)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-04-16 160033](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/97d31256-7696-4e86-b34a-4073ae4692e1)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![Screenshot 2024-04-16 160046](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/4132c821-4f54-4678-a01e-b01cc5d54f41)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![Screenshot 2024-04-16 160057](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/0cce662a-5833-4f25-8e75-c9dd61e1bd7f)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![Screenshot 2024-04-16 161236](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/bebd0c95-b10e-47c9-909b-487f33678cab)

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![Screenshot 2024-04-16 161250](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/c7d4a5af-168e-4370-aa9a-7e44d04ba693)

```
data.isnull().sum()
```
![Screenshot 2024-04-16 161257](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/c5796fa4-60dd-4944-8324-3c815fbb6201)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-04-16 161347](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/1f876928-fc3b-4c8c-83d7-b82997817c76)

```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-04-16 161405](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/4a8053cb-ada5-4e5c-b796-36454c312fa2)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-04-16 161416](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/66c8854e-fdea-4c46-86d0-7b125f914775)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-04-16 161423](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/1e0bec0c-8140-428a-823f-bb1376907197)

```
data2
```
![Screenshot 2024-04-16 161446](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/7b85e260-cd12-4ab3-b053-2d77282fc626)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![Screenshot 2024-04-16 161454](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/55201f32-04cb-4690-9380-14913587dbe7)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-04-16 161516](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/c6e2efe8-5da4-44d7-aa02-4205f2dcf40e)

```
features = list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-04-16 161524](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/296d5e9f-9bad-486a-9e81-f35a1f104098)

```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-04-16 161529](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/beec4467-82a6-4ab9-8907-e026ab4ed831)

```
x=new_data[features].values
print(x)
```
![Screenshot 2024-04-16 161534](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/d66dfef0-d20b-4dd7-aa3f-f204f933d51c)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]

}
df=pd.DataFrame(data)
df
```
![Screenshot 2024-04-16 161556](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/992a61fe-e9c6-4a77-b71a-5fa4af4c9ebb)

```
X=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
```
![Screenshot 2024-04-16 161604](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/f741be11-a996-44d4-ae06-32ebeaf37caf)

```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-04-16 161610](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/3dd0cb8a-7fc2-4272-9b1b-eb99b35c9550)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-04-16 161616](https://github.com/gokulapriya632202/EXNO-4-DS/assets/119560302/55288e80-e6d6-4d55-bb50-4012850e493a)










# RESULT:
       # INCLUDE YOUR RESULT HERE
