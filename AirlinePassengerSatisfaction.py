#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download
#louis boulas 6/2023


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df1 = pd.read_csv(r'E:\CS_DS_Projects\Datasets\AirlinePassengerSatisfaction\train.csv')
df_test = pd.read_csv(r'E:\CS_DS_Projects\Datasets\AirlinePassengerSatisfaction\test.csv')


# # Data Exploration and Cleaning

# In[4]:


pd.set_option('display.max_columns', None)
df1.head()


# In[5]:


df1.shape


# In[6]:


df1 = df1.drop(['Unnamed: 0','id'], axis=1)
df_test = df_test.drop(['Unnamed: 0', 'id'], axis=1)


# In[7]:


df1.info()


# In[8]:


#Study of Characteristics
#describing columns
for column in df1:
    print(df1[column].describe())


# In[9]:


#checking for null values
df1.isnull().sum() 


# In[10]:


df1.info()


# Since there is less than 1% of the data that has null values and they are all located in Arrival Delay, I am opting to drop those rows. Another strategy would be to replace the null values with the median of the column.

# In[11]:


df1 = df1.dropna()
df_test =  df_test.dropna()


# In[12]:


df1.isnull().any()


# In[13]:


df1 = df1.rename(columns={"Departure Delay in Minutes":"Departure Delay", "Arrival Delay in Minutes":"Arrival Delay", "Leg room service":"Leg Room","Inflight wifi service":"Wifi"})
df_test = df_test.rename(columns={"Departure Delay in Minutes":"Departure Delay", "Arrival Delay in Minutes":"Arrival Delay", "Leg room service":"Leg Room","Inflight wifi service":"Wifi"})


# ### Finding and Cleaning Data Outliers

# In[14]:


def plot(df,whis):
    for column in df:
        df[[column]].plot.box(figsize=(10,5), title= column, whis = whis)


# In[15]:


#picking out numerical columns for box plots
df2 = df1.drop(['Gender','Customer Type', 'Type of Travel','Class','satisfaction'],axis=1)


# In[16]:


plot(df2,1.5)


# In[17]:


#outliers found in: flight distance, checkin service, departure delay, arrival delay
plot(df2,2)


# In[18]:


#IQR bounds function
def check_outliers(x,factor):
    q25 = x.quantile(.25)
    q75 = x.quantile(.75)
    IQR = q75-q25
    l = q25 - IQR*factor
    u = q75 + IQR*factor
    return l, u


# In[19]:


lower, upper = check_outliers(df2['Checkin service'],1.5)
print(f'lower = {lower}, upper={upper}')


# In[20]:


sns.histplot(df1['Flight Distance'], bins=50, color='green', kde =True)


# In[21]:


df1['ew_bin'] = pd.cut(df1['Flight Distance'], 3)
df1['ew_bin'].value_counts().sort_index()


# In[22]:


df1['q_bin'] = pd.qcut(df1['Flight Distance'], 3)
df1['q_bin'].value_counts().sort_index()


# Going to bin flight distance values into the following categories:
# short haul <700mi, medium haul 700mi - 3000mi, long haul >3000mi
# 
# source: __[United definition from wikipedia](https://en.wikipedia.org/wiki/Flight_length#:~:text=American%20Airlines%20defines%20short%2D%2F,New%20York%E2%80%93San%20Francisco%20routes)__

# In[23]:


df1['FD_bin'] = pd.cut(x=df1['Flight Distance'], bins=[0,700,3000,5000], labels=['short haul', 'medium haul', 'long haul'])
df1 = df1.drop(['Flight Distance'], axis=1)

df_test['FD_bin'] = pd.cut(x=df_test['Flight Distance'], bins=[0,700,3000,5000],labels=['short haul', 'medium haul',
                                                                                        'long haul'])
df_test = df_test.drop(['Flight Distance'], axis=1)


# In[24]:


df1['FD_bin'].value_counts().sort_index()


# In[25]:


df1.drop(['ew_bin','q_bin'], axis=1)


# Taking a deeper look into Departure Delay and Arrival Delay

# In[26]:


#DEPARTURE/ARRIVAL DELAY
sns.histplot(df1['Departure Delay'], bins=10, color='green', kde =True)


# In[27]:


sns.histplot(df1['Arrival Delay'], bins=10, color='green', kde =True)


# In[28]:


df1[['Departure Delay','Arrival Delay']].plot( color = ['green', 'brown'])


# From the graphs above we can see that Departure Delay and Arrival Delay are just about the same. In other words, if a flight is delayed pre departure then it will definitely have an arrival delay as well. Therefore, we will drop Departure Delay from our data and only consider Arrival Delay since having both would be redundant.

# In[29]:


df1 = df1.drop(['Departure Delay'], axis=1)
df_test = df_test.drop(['Departure Delay'], axis=1)


# In[30]:


df1 = df1.drop(['ew_bin','q_bin'],axis=1)
df1.info()


# In[31]:


df1.groupby('satisfaction')['Arrival Delay'].mean()


# In[32]:


#change outliers to upper bound
lower, upper = check_outliers(df1['Arrival Delay'],1.5)
print(f'lower = {lower}, upper={upper}')


# In[33]:


#number of outliers
mask_upper = df1['Arrival Delay']>upper
mask_upper.sum()


# In[34]:


for i in df1[mask_upper].index:
  df1.loc[i,'Arrival Delay']=upper


# In[35]:


#checking if all outliers in Arrival Delay were handled.
new_mask_upper = df1['Arrival Delay']>upper
new_mask_upper.sum()


# In[36]:


df1[['Arrival Delay']].plot.box(figsize=(10,5), title= "Arrival Delay IQR Plot", whis = 1.5)


# ### Further Data Exploration

# In[37]:


satis_counts = df1['satisfaction'].value_counts()
labels = satis_counts.index
values = satis_counts.values

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title('customer satisfaction')
plt.show()


# The number of satisfied vs neutral or dissatisfied customers within the dataset is approximately the same. This will be our target variable when we run our models later.

# In[38]:


fig, axes = plt.subplots(3, 2, figsize = (12,12))
sns.countplot(data=df1, x='Gender', hue= 'satisfaction', ax = axes[0,0])
sns.countplot(data=df1, x='Customer Type', hue= 'satisfaction', ax = axes[0,1])
sns.countplot(data=df1, x='Type of Travel', hue= 'satisfaction', ax = axes[1,0])
sns.countplot(data=df1, x='Class', hue= 'satisfaction', ax = axes[1,1])
sns.countplot(data=df1, x='FD_bin', hue= 'satisfaction', ax = axes[2,0])
sns.histplot(data=df1, x='Age', hue= 'satisfaction', ax = axes[2,1])


# In[39]:


df1.groupby('satisfaction')['Class'].describe()


# In[40]:


df1.groupby('Class')['satisfaction'].describe()


# In[41]:


df1.groupby('FD_bin')['satisfaction'].describe()


# In[42]:


df1.groupby('satisfaction')['Age'].describe()


# In[43]:


df1.groupby('Type of Travel')['satisfaction'].describe()


# In[44]:


df1.groupby('Wifi')['satisfaction'].describe()


# ### Key Takeaways
# -  There is an even split between male and female customers and they both have about the same proportion of being satisfied vs neutral or unsatisfied
# -  Our dataset has many more loyal customers than disloyal customers, meaning the customers in our data are ones that keep using this particular airline
# -  The majority of our data comprises of clients who fly for business purposes
# -  About half of the data consists of business class flights where about two thirds of those in business class are satisfied
# -  The longer the flight the greater satisfaction ratio: long haul > medium haul > short haul (satisfied clients / total clients)
# -  The age group of 40-60 is more satisfied whereas every else has higher number of neutral or dissatisfied clients

# In[45]:


print(df1.corr())


# In[46]:


sns.heatmap(df1.corr(), cmap= 'YlGnBu')

plt.show()


# # Modelling
# 
# ### Model Selection
# 
# The goal of this data mining project is to find out which features are the most critical in determining a customers satisfaction with their flight. This includes features about the customer themselves, and the flight they took. This can be boiled down to a classification problem where we want to predict whether a customer will be satisfied or neutral/dissatisfied with their trip (binary outcome).
# 
# Thus, I am choosing to implemnent 3 different models. The first one being a **Logistic Regression Model**, which will the baseline model for our data set. Then I will be using a **Decision Tree Classifier** and a **Random Forest Classifier**. The reasoning behind using these models is because the latter two easily evaluate variable importance/contribution to the model. Since I am less interested in the actual outcome and most interested in which features lead to the outcome, these models will be able to paint a robust picture of what I am aiming for.

# In[47]:


#model packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report


# In[48]:


df_test.info()


# ## Preprocessing
# ### Using pd.get_dummies to turn categorical columns to numerical

# In[49]:


#preprocessing categorical columns using pd.get_dummies and preperaring both
#training and testing datasets for modelling
train_features = df1.drop(['satisfaction'], axis=1)
X_train = pd.get_dummies(train_features, columns = ['Gender','Customer Type','Type of Travel',
                                                'Class','FD_bin'])
y_train = df1['satisfaction']

test_features = df_test.drop(['satisfaction'], axis=1)
X_test = pd.get_dummies(test_features, columns = ['Gender','Customer Type','Type of Travel',
                                                'Class','FD_bin'])
y_test = df_test['satisfaction']


# In[50]:


train_features.info()


# ## LOGISTIC REGRESSION

# In[51]:


#LOGISTIC REGRESSION
#using random state 42 for reproducable results


# In[52]:


#define model
logreg = LogisticRegression(random_state=42, max_iter = 1000)

#fit model
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

#Accuracy, precision, and recall scores
acc_score = accuracy_score(y_test,y_pred)
print('accuracy:', acc_score)

p0 = precision_score(y_test,y_pred, average="binary", pos_label="neutral or dissatisfied")
print('class 0 precision:', p0)
p1 = precision_score(y_test,y_pred, average="binary", pos_label="satisfied")
print('class 1 precision:', p1)

r0 = recall_score(y_test,y_pred, average="binary", pos_label="neutral or dissatisfied")
print('class 0 recall:', r0)
r1 = recall_score(y_test,y_pred, average="binary", pos_label="satisfied")
print('class 1 recall:', r1)


# In[53]:


print(classification_report(y_test,y_pred))


# In[54]:


#find out the relative importance of each columns
X_columns = X_test.columns
importance = logreg.coef_[0]
#plotting it:
plt.figure(figsize=(9,4))
plt.bar(X_columns, importance)
plt.xticks(rotation=90)
plt.title('bar chart of logistic regression coefficients as feature importance scores')
plt.show()


# #### Features Importance in Logistic Regression
# 
# -  Online Boarding is the most important feature for determining which customers are going to be satisfied followed by Wifi
# -  Whereas Type of Travel Personal Travel and Customer Type Disloyal Customer are strong indicators for predicting which customers are going to be dissatisfied

# ## Decision Tree Classifier

# In[55]:


#DECISION TREE

dtc = DecisionTreeClassifier(random_state=42)

dtc.fit(X_train,y_train)
y_pred_dtc = dtc.predict(X_test)

acc_score = accuracy_score(y_test,y_pred_dtc)
print('accuracy_score:', acc_score)

p0 = precision_score(y_test,y_pred_dtc, average="binary", pos_label="neutral or dissatisfied")
print('class 0 precision:', p0)
p1 = precision_score(y_test,y_pred_dtc, average="binary", pos_label="satisfied")
print('class 1 precision:', p1)

r0 = recall_score(y_test,y_pred_dtc, average="binary", pos_label="neutral or dissatisfied")
print('class 0 recall:', r0)
r1 = recall_score(y_test,y_pred_dtc, average="binary", pos_label="satisfied")
print('class 1 recall:', r1)


# In[56]:


print(classification_report(y_test,y_pred_dtc))


# In[57]:


importance = dtc.feature_importances_

# plot feature importance
plt.figure(figsize=(8,6))
plt.bar(X_columns, importance)
plt.title('Bar Chart of Decision Tree Classifier Coefficients as Feature Importance Scores')
plt.xticks(rotation=90)
plt.show()


# #### Features Importance in Decision Tree Classifier
# 
# -  Online boarding and Wifi are again the top two features of importance
# -  However, with this model we can also see the importance of the Type of Travel, specifically that type being of value Personal Travel

# ## RANDOM FOREST CLASSIFIER

# In[58]:


#RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier(random_state=42)

rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

#accuracy, precison and recall score
acc_score = accuracy_score(y_test,y_pred_rfc)
print('accuracy_score:', acc_score)

p0 = precision_score(y_test,y_pred_rfc, average="binary", pos_label="neutral or dissatisfied")
print('class 0 precision:', p0)
p1 = precision_score(y_test,y_pred_rfc, average="binary", pos_label="satisfied")
print('class 1 precision:', p1)

r0 = recall_score(y_test,y_pred_rfc, average="binary", pos_label="neutral or dissatisfied")
print('class 0 recall:', r0)
r1 = recall_score(y_test,y_pred_rfc, average="binary", pos_label="satisfied")
print('class 1 recall:', r1)


# In[59]:


print(classification_report(y_test,y_pred_rfc))


# In[60]:


importance = rfc.feature_importances_

# plot feature importance
plt.figure(figsize=(8,6))
plt.bar(X_columns, importance)
plt.xticks(rotation=90)
plt.title('Bar Chart of Random Forest Classifier Coefficients as Feature Importance Scores')
plt.show()


# #### Features Importance in Logistic Regression
# 
# -  Again, Online Boarding and Wifi are clearly the top 2 features
# -  We can also see that both the Type of Travel and Class Business play an important role in predicting satisfaction
# -  Closely following those we also have seat comfort

# ### Insights, Conclusions, and Future Improvements
# 
# From the models we can see that the Random Forest Classifier has the highest accuracy, precision and recall score. I would thus suggest that for our purposes of trying to find out which features are of utmost importance for a customers satisfaction that we use the Random Forest Classifier as our main model.
# 
# I'd suggest the airline to focus on providing the best experience when in checking in online/in advance through the web. The development of an easy to use app would be critical in the world we live in today and would be the primary driver behind customer satisfaction. Secondly, I would suggest offering some sort of wifi support on all aircrafts if possible. Whether it be free or paid wifi as this dataset does not make a distinction between the two. Thidly, the logistic regression model tells us the importance of customer loyalty. Customers who are "disloyal" are more likely to be neutral or unsatisfied and thus an airline should look for additional ways to attract those customers to their loyalty program.
# 
# Additionally, I'd suggest looking into gathering more data about the differences between business and economy that sway customers satisfaction. Is it customer service? Is it seat comfort? This dataset doesn't paint a clear enough picture of where an airline could improve in bridging the gap of customer satisfaction between personal vs business travel and between business vs eco class.
# 
# In terms of improvements that can be done, I would consider implementing dimensionality reduction in the data pre-processing step. This path would allow us to really focus on the features of importance because, as we have seen, many of the independent variables did not play a significant part in any of our models. For example, Gender, Flight Distance, Delays, etc. This would allow us to better fine tune our models and really only focus on the features that matter in determining a customers satisfaction.
