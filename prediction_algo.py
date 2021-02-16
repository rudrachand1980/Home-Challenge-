##importing the data and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#Importing datasets
train = pd.read_csv('2021-01-21_zeta-disease_training-data_dsi-take-home-challenge.csv')
test = pd.read_csv('2021-01-21_zeta-disease_prediction-data_dsi-take-home-challenge.csv')
train.head()
test.head()
#remove the response column
test.drop(['zeta_disease'], axis=1,inplace=True)
##missing data in train and test
train.isnull().sum()
##missing data in test set
test.isnull().sum()
#There is no missing values in datasets
#To check multicolinearity between independent variables
plt.figure(figsize=(30, 20))
plt.title('Correlation between variables')
sns.heatmap(test.corr(), annot=True, cmap='Blues');
#years of smoking and age , is higher colinearity
#weight and insulin_test has higher colinearity
#But the colinearity is not that much which we can consider, mostly independent variables are not mutually correlated
# feature meatrix and response vector seperation
X_train=train.iloc[:,0:-1]
y_train=train['zeta_disease']
X_train.head()
#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.25)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#Model building
#Using Logistic regression
logit_model = LogisticRegression(penalty='l2')
clf_logit = logit_model.fit(X_train,y_train)
y_pred=clf_logit.predict(X_test)
print(accuracy_score(y_test,y_pred))
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = clf_logit.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#Confusion Matrix Logistic Regression
plt.rcParams['figure.figsize'] = (5, 5)
cm_logit = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_logit, annot = True, cmap = 'Greens')
plt.show()
# classification report
cr = classification_report(y_test, y_pred)
print(cr)
# Using a Decision Tree classifier 
from sklearn.tree import DecisionTreeClassifier
param_grid={'max_depth':range(1,20,2)}
DT=DecisionTreeClassifier()
clf_DT=GridSearchCV(DT,param_grid,cv=10,scoring='accuracy',n_jobs=-1).fit(X_train,y_train)
y_pred=clf_DT.predict(X_test)
print(accuracy_score(y_test,y_pred))
#Using a Random Forest tree classifier
from sklearn.ensemble import RandomForestClassifier
param_grid={'max_depth':range(1,20,2)}
RF=RandomForestClassifier()
clf_rf=GridSearchCV(RF,param_grid,cv=10,scoring='accuracy',n_jobs=-1).fit(X_train,y_train)
y_pred=clf_rf.predict(X_test)
accuracy_score(y_test,y_pred)
#Lets use k-fold cross validation
from sklearn.model_selection import KFold,cross_val_score
k_fold = KFold(n_splits=10,shuffle=True,random_state=0)
score= cross_val_score(DT,X_train,y_train,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)
print(f'average score Decission Tree : {round(np.mean(score),2)}')
score= cross_val_score(RF,X_train,y_train,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)
print(f'average score Random Forest : {round(np.mean(score),2)}')
clf=SVC()
score = cross_val_score(clf,X_train,y_train,cv=k_fold,n_jobs=-1,scoring='accuracy')
print(score)
print(f'average score Support Vector : {round(np.mean(score),2)}')
##Applying Sampling Technique
#just to see the ratio of the category under the response variable
train['zeta_disease'].value_counts()
# Sampling Techniques are used to balance the target variable which are not in same samples
# There are several sampling techniques
# 1)Under Sampling
# 2)Up sampling
# 3)SMOT  ==== Let's apply this and see the accuracy
# 4)Class Weight
#Up sampling with SMOTE
from imblearn.over_sampling import SMOTE
x_resample, y_resample  = SMOTE().fit_sample(X_train, y_train.values.ravel())

print(x_resample.shape)
print(y_resample.shape)
from sklearn.model_selection import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)
print(x_train2.shape)
print(y_train2.shape)
print(x_test2.shape)
print(y_test2.shape)
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model_o = RandomForestClassifier()
model_o.fit(x_train2, y_train2)

y_pred = model_o.predict(x_test2)

print("Training Accuracy: ", model_o.score(x_train2, y_train2))
print('Testing Accuarcy: ', model_o.score(x_test2, y_test2))

# confusion matrix
cm = confusion_matrix(y_test2, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, cmap = 'winter')
plt.show()
# classification report
cr = classification_report(y_test2, y_pred)
print(cr)
#Final Model #Final Model  clf_DT gives better result
predictions = model_o.predict(test).tolist()
test['zeta_disease'] = predictions
test.to_csv('zeta-disease_prediction-data.csv', index = False)
test.head()