# Predictive Analysis: Electric Vehicle Interest

*Load data*


```python
import pandas as pd
import numpy as np

test_df = pd.read_csv("./test.csv")
train_df = pd.read_csv("./train.csv")
```

*Preview the first few rows*


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Age</th>
      <th>Annual_Income_USD</th>
      <th>Daily_Commute_km</th>
      <th>Number_of_Cars_Owned</th>
      <th>Charging_Stations_Near_Home</th>
      <th>Charging_Stations_Near_Work</th>
      <th>Environmental_Concern_Level</th>
      <th>Gender</th>
      <th>City_Type</th>
      <th>Current_Car_Type</th>
      <th>Home_Charging_Possible</th>
      <th>Subsidy_Available</th>
      <th>Range_Anxiety_Level</th>
      <th>Will_Buy_EV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>66</td>
      <td>92887.0</td>
      <td>23.4</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>1.0</td>
      <td>Male</td>
      <td>Suburban</td>
      <td>Sedan</td>
      <td>Yes</td>
      <td>No</td>
      <td>Low</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38</td>
      <td>30000.0</td>
      <td>5.0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4.0</td>
      <td>Male</td>
      <td>Rural</td>
      <td>SUV</td>
      <td>Yes</td>
      <td>No</td>
      <td>Low</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>26</td>
      <td>94389.0</td>
      <td>36.8</td>
      <td>1</td>
      <td>8</td>
      <td>15</td>
      <td>5.0</td>
      <td>Female</td>
      <td>Urban</td>
      <td>Sedan</td>
      <td>No</td>
      <td>Yes</td>
      <td>Low</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>66</td>
      <td>73580.0</td>
      <td>23.7</td>
      <td>2</td>
      <td>6</td>
      <td>9</td>
      <td>3.0</td>
      <td>Male</td>
      <td>Suburban</td>
      <td>Hatchback</td>
      <td>Yes</td>
      <td>No</td>
      <td>Low</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>54</td>
      <td>57898.0</td>
      <td>50.8</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3.0</td>
      <td>Male</td>
      <td>Suburban</td>
      <td>Hatchback</td>
      <td>Yes</td>
      <td>No</td>
      <td>Low</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



*Information on data types*


```python
train_df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 668665 entries, 0 to 668664
    Data columns (total 15 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   id                           668665 non-null  int64  
     1   Age                          668665 non-null  int64  
     2   Annual_Income_USD            668665 non-null  float64
     3   Daily_Commute_km             668665 non-null  float64
     4   Number_of_Cars_Owned         668665 non-null  int64  
     5   Charging_Stations_Near_Home  668665 non-null  int64  
     6   Charging_Stations_Near_Work  668665 non-null  int64  
     7   Environmental_Concern_Level  668665 non-null  float64
     8   Gender                       668665 non-null  str    
     9   City_Type                    668665 non-null  str    
     10  Current_Car_Type             668665 non-null  str    
     11  Home_Charging_Possible       668665 non-null  str    
     12  Subsidy_Available            668665 non-null  str    
     13  Range_Anxiety_Level          668665 non-null  str    
     14  Will_Buy_EV                  668665 non-null  str    
    dtypes: float64(3), int64(5), str(7)
    memory usage: 93.5 MB
    

*Let's examine the distribution of our target variable `Will_Buy_EV`.*


```python
target_counts = train_df["Will_Buy_EV"].value_counts()
target_distribution = train_df["Will_Buy_EV"].value_counts(normalize=True)
print(f"target_counts :\n{target_counts}\n")
print(f"\ntarget_distribution :\n{target_distribution}")
```

    target_counts :
    Will_Buy_EV
    No     551886
    Yes    116779
    Name: count, dtype: int64
    
    
    target_distribution :
    Will_Buy_EV
    No     0.825355
    Yes    0.174645
    Name: proportion, dtype: float64
    

### Finding the best features for our model
We use concepts from Information Theory (entropy, conditional entropy, information gain) to identify which features are most predictive.

#### **Entropy**:
$$H(X) = -\sum_{x \in X} p(x) \log_2 p(x)$$

#### **Joint Entropy**:
$$H(X, Y) = -\sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2 p(x, y)$$

#### **Conditional Entropy**:
$$H(X|Y) = H(X,Y) - H(Y)$$

#### **Information Gain**:
$$I(X, Y) = H(X) - H(X | Y)$$

---
Function Implementation 


```python
def entropy(X):
    X = X/np.sum(X)
    # Add 1e-12 (epsilon) for numerical stability (avoid log(0))
    return -(np.sum(X*np.log2(X + 1e-12)))

def joint_entropy(X,Y):
    joint_counts = pd.crosstab(X, Y)
    return entropy(joint_counts.values.flatten())

def conditional_entropy(X,Y):
    joint = joint_entropy(X,Y)
    return joint - entropy(Y.value_counts().values)

def information_gain(df, target_col, feature_col):
    h_target = entropy(df[target_col].value_counts().values)
    h_cond = conditional_entropy(df[target_col], df[feature_col])
    return h_target - h_cond

```

---
Calculate Information Gain for all features


```python
features = train_df.columns.drop(["Will_Buy_EV", "id"])
results = {
    col: round(information_gain(train_df, "Will_Buy_EV", col), 3)
    for col in features
}
sorted_i = pd.Series(results).sort_values(ascending=False)
features_to_keep = sorted_i[sorted_i > 0.003].index.tolist()
print("Feature Ranking by Information Gain:")
print(sorted_i)
```

    Feature Ranking by Information Gain:
    Environmental_Concern_Level    0.182
    Subsidy_Available              0.117
    Annual_Income_USD              0.090
    Range_Anxiety_Level            0.013
    Daily_Commute_km               0.007
    Home_Charging_Possible         0.005
    Age                            0.003
    City_Type                      0.001
    Charging_Stations_Near_Home    0.001
    Number_of_Cars_Owned           0.000
    Charging_Stations_Near_Work    0.000
    Gender                         0.000
    Current_Car_Type               0.000
    dtype: float64
    

Feature Selection

We used Information Gain to identify the most predictive features. To balance model accuracy and simplicity, we applied a threshold of 0.003, retaining only the features that contribute significantly to the target variable.

Selected Features (Gain > 0.003):

    Environmental_Concern_Level (0.182)

    Subsidy_Available (0.117)

    Annual_Income_USD (0.090)

    Range_Anxiety_Level (0.013)

    Daily_Commute_km (0.007)

    Home_Charging_Possible (0.005)

---
We can now start to build our predictive model using these features


```python
from sklearn.model_selection import train_test_split
X = train_df[features_to_keep]
y = train_df["Will_Buy_EV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

```


```python
from catboost import CatBoostClassifier

cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
model = CatBoostClassifier(cat_features=cat_features)
model.fit(X_train, y_train)
```



```python
from sklearn.metrics import roc_auc_score


prediction = model.predict_proba(X_test)[:,1]
score = roc_auc_score(y_test, prediction)
print(f"AUC : {score*100:.2f}%")
```

    AUC : 94.15
    


```python
import matplotlib.pyplot as plt

plt.barh(features_to_keep, model.feature_importances_)
plt.xlabel("Importance")
plt.show()
```


    
![png](main_files/main_19_0.png)
    


The feature importance from the model confirms our initial information gain calculation; however, we can see that `Environmental_Concern_Level` is less important than `Subsidy_Available`, even though it had the highest information gain.

---
We can now use our model on the test dataset.


```python
prediction = model.predict_proba(test_df[features_to_keep])[:,1]
submission = pd.DataFrame({
    "id":test_df["id"],
    "Will_Buy_EV":prediction
})
submission.to_csv("submission.csv",index=False)
```
