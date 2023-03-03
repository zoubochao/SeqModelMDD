import datetime
import pickle
from time import time

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = pickle.load(open('X_phone_ring_date_28.pkl', 'rb'))
y = pickle.load(open('y_phone_ring_28.pkl', 'rb'))

for column in list(X.columns[X.isnull().sum() > 0]):
    mean_val = X[column].mean()
    X[column].fillna(mean_val, inplace=True)
# X = X.fillna(value=0)

print(X)
print(y)

all_recordID = y.iloc[:,0]

train_set_ids, test_set_ids = train_test_split(all_recordID, test_size=30, random_state=1)

train_set = X[X['Id'].isin(train_set_ids)]
test_set = X[X['Id'].isin(test_set_ids)]

train_set = train_set.drop('Id', axis=1)
test_set = test_set.drop('Id', axis=1)

col = train_set.columns.tolist()
# col.remove('date')
# col.remove('HAMD')
# print(col)

ss = StandardScaler()
ss = ss.fit(train_set.loc[:,col])
train_set.loc[:,col] = ss.transform(train_set.loc[:,col])
test_set.loc[:,col] = ss.transform(test_set.loc[:,col])

train_X = train_set.values.reshape(len(train_set_ids), -1)
test_X = test_set.values.reshape(len(test_set_ids), -1)

train_y = y[y.iloc[:,0].isin(train_set_ids)]
test_y = y[y.iloc[:,0].isin(test_set_ids)]

train_y = train_y.iloc[:,1].values.ravel()
test_y = test_y.iloc[:,1].values.ravel()

train_y = train_y.astype('int')
test_y = test_y.astype('int')

X_ = X
y_ = y
X_ = X_.drop('Id', axis = 1)
y_ = y_.iloc[:, 1].values.ravel()
X_.loc[:,col] = ss.transform(X_.loc[:,col])
X_ = X_.values.reshape(len(y_), -1)
y_ = y_.astype('int')

print(X_.shape)
print(y_.shape)

svm_m = svm.SVC()
svm_p = cross_val_score(svm_m, X_, y_, cv=10, scoring='average_precision')
svm_r = cross_val_score(svm_m, X_, y_, cv=10, scoring='recall')
svm_f = cross_val_score(svm_m, X_, y_, cv=10, scoring='f1')
svm_a = cross_val_score(svm_m, X_, y_, cv=10, scoring='roc_auc')
print(svm_p.mean(), svm_r.mean(), svm_f.mean(), svm_a.mean())

rfc = RandomForestClassifier()
rfc_p = cross_val_score(rfc, X_, y_, cv=10, scoring='average_precision')
rfc_r = cross_val_score(rfc, X_, y_, cv=10, scoring='recall')
rfc_f = cross_val_score(rfc, X_, y_, cv=10, scoring='f1')
rfc_a = cross_val_score(rfc, X_, y_, cv=10, scoring='roc_auc')
print(rfc_p.mean(), rfc_r.mean(), rfc_f.mean(), rfc_a.mean())

gnb = GaussianNB()
gnb_p = cross_val_score(gnb, X_, y_, cv=10, scoring='average_precision')
gnb_r = cross_val_score(gnb, X_, y_, cv=10, scoring='recall')
gnb_f = cross_val_score(gnb, X_, y_, cv=10, scoring='f1')
gnb_a = cross_val_score(gnb, X_, y_, cv=10, scoring='roc_auc')
print(gnb_p.mean(), gnb_r.mean(), gnb_f.mean(), gnb_a.mean())

lr = LogisticRegression()
lr_p = cross_val_score(lr, X_, y_, cv=10, scoring='average_precision')
lr_r = cross_val_score(lr, X_, y_, cv=10, scoring='recall')
lr_f = cross_val_score(lr, X_, y_, cv=10, scoring='f1')
lr_a = cross_val_score(lr, X_, y_, cv=10, scoring='roc_auc')
print(lr_p.mean(), lr_r.mean(), lr_f.mean(), lr_a.mean())
