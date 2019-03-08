#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:11:15 2019

@author: ed
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:36:12 2019

@author: User
"""
#%%
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import precision_score, accuracy_score, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

file_2014='/home/ed/Downloads/ny_SPARCS_discharges_2014.csv'
#replace with scraper
file_2015='/home/ed/Downloads/ny_SPARCS_discharges_2015.csv'

sparcs_2015=pd.read_csv(file_2015, engine='python')
sparcs_2014=pd.read_csv(file_2015, engine='python')


def clean(data):
    data.columns = data.columns.str.strip().str.upper().str\
    .replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return data

sparcs_2015=clean(sparcs_2015)
sparcs_2014=clean(sparcs_2014)
#2015
numeric_cols = [cname for cname in sparcs_2015.columns if 
                                sparcs_2015[cname].dtype in ['int64', 'float64']]
for columns in numeric_cols:
    sparcs_2015[columns].fillna(0, inplace=True)
    
text_cols = [cname for cname in sparcs_2015.columns if 
                                sparcs_2015[cname].dtype not in ['int64', 'float64']]

for columns in text_cols:
    sparcs_2015[columns].fillna("NA", inplace=True)

#2014
numeric_cols = [cname for cname in sparcs_2014.columns if 
                                sparcs_2014[cname].dtype in ['int64', 'float64']]
for columns in numeric_cols:
    sparcs_2014[columns].fillna(0, inplace=True)
    
text_cols = [cname for cname in sparcs_2014.columns if 
                                sparcs_2014[cname].dtype not in ['int64', 'float64']]

for columns in text_cols:
    sparcs_2014[columns].fillna("NA", inplace=True)


#%%
le = LabelEncoder()
sparcs_2015['APR_RISK_OF_MORTALITY']=sparcs_2015['APR_RISK_OF_MORTALITY'].replace("NA","Minor")
le.fit(["Minor", "Moderate", "Major", "Extreme"])
y=le.transform(sparcs_2015['APR_RISK_OF_MORTALITY'])


equiv_mort = {'Minor':0, 'Moderate':1, 'Major':0, 'Extreme': 0}
sparcs_2015['y_binary']=sparcs_2015['APR_RISK_OF_MORTALITY'].map(equiv_mort)
y_binary=sparcs_2015['y_binary'].values

proposed_X = sparcs_2015.drop(['ATTENDING_PROVIDER_LICENSE_NUMBER','OPERATING_PROVIDER_LICENSE_NUMBER',
                               'OPERATING_CERTIFICATE_NUMBER','OTHER_PROVIDER_LICENSE_NUMBER',
                               'TOTAL_CHARGES','TOTAL_COSTS','APR_RISK_OF_MORTALITY',
                               'DISCHARGE_YEAR',
                               'CCS_PROCEDURE_CODE','APR_DRG_CODE',
                               'CCS_DIAGNOSIS_CODE',
                               'APR_MDC_CODE','APR_SEVERITY_OF_ILLNESS_CODE',
                               'APR_MEDICAL_SURGICAL_DESCRIPTION',
                               'APR_SEVERITY_OF_ILLNESS_DESCRIPTION',
                               'APR_MDC_DESCRIPTION',
                               'APR_DRG_DESCRIPTION',
                               'PATIENT_DISPOSITION',
                                'PAYMENT_TYPOLOGY_1',
                                 'PAYMENT_TYPOLOGY_2',
                                 'PAYMENT_TYPOLOGY_3','y_binary',
                                 'BIRTH_WEIGHT'],axis=1)

low_cardinality_cols = [cname for cname in proposed_X.columns if 
                                proposed_X[cname].nunique() < 20 and
                                proposed_X[cname].dtype == "object"]
numeric_cols = [cname for cname in proposed_X.columns if 
                                proposed_X[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

# One hot encoded
one_hot_encoded_X = pd.get_dummies(proposed_X[my_cols])
print("# of columns after one-hot encoding: {0}"\
      .format(len(one_hot_encoded_X.columns)))
one_hot_encoded_X=clean(one_hot_encoded_X)

features=list(one_hot_encoded_X.columns)

#%%
X_train, X_test, y_train, y_test =train_test_split(one_hot_encoded_X, 
                                                   y, test_size=0.2, 
                                                   random_state=4)

dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm',
                         feature_names=features)
dtest_svm = xgb.DMatrix('dtest.svm',
                        feature_names=features)

tests={}
param = {
    'max_depth':20,  # the maximum depth of each tree
    'eta': 0.5,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softmax',  
    'num_class': 4 # the number of classes that exist in this datset
    }  
#param['eval_metric'] = ['auc']
num_round = 50  # the number of training iterations
watchlist = [(dtest_svm,'eval'), (dtrain_svm,'train')]



evals_result = {}
bst = xgb.train(param, dtrain_svm, 
                num_round, watchlist, 
                evals_result=evals_result,
                early_stopping_rounds=2)

bst.dump_model('/home/ed/Desktop/nysparcdischarges/dump.raw.txt')

preds = bst.predict(dtest_svm)
y_score=preds
best_preds = np.asarray([np.argmax(line) for line in preds])


joblib.dump(bst, '/home/ed/Desktop/nysparcdischarges/bst_model.pkl', compress=True)
# bst = joblib.load('bst_model.pkl') # load it later
#metrics.f1_score(y_test, y_score, average='weighted', labels=np.unique(y_score))

xgb.plot_importance(bst, importance_type='weight',max_num_features=10)

xgb.to_graphviz(bst, rankdir='lr',size="20,20!")

accuracy_score(y_test, y_score)
#%%
X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_X, y_binary, test_size=0.2, random_state=4)

xgtrain = xgb.DMatrix(X_train, label=y_train)
clf = xgb.XGBClassifier(missing=np.nan,
                max_depth = 7,
                n_estimators=700,
                learning_rate=0.1, 
                nthread=4,
                subsample=1.0,
                colsample_bytree=0.5,
                min_child_weight = 3,
                seed=1301)
xgb_param = clf.get_xgb_params()
#do cross validation
print ('Start cross validation')
cvresult=xgb.cv(xgb_param, xgtrain, num_boost_round=80, nfold=3, 
                stratified=True, folds=None, metrics=(), obj=None, 
                feval=None, maximize=False, early_stopping_rounds=2, 
                fpreproc=None, as_pandas=True, verbose_eval=None, 
                show_stdv=True, seed=0, callbacks=None, shuffle=True)

#sklearn plotting
print('Best number of trees = {}'.format(cvresult.shape[0]))
clf.set_params(n_estimators=cvresult.shape[0])
print('Fit on the training data')
clf.fit(X_train, y_train, eval_metric='auc')
print('Overall AUC:', roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]))
print('Predict the probabilities on test set')
pred = clf.predict_proba(X_train, ntree_limit=cvresult.shape[0])

joblib.dump(clf, '/home/ed/Desktop/nysparcdischarges/clf_model.pkl', compress=True)
# bst = joblib.load('bst_model.pkl') # load it later

xgb.plot_importance(clf)

xgb.to_graphviz(clf,size="20,20!")


y_pred_xgb = clf.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_xgb)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

average_precision = average_precision_score(y_test, y_pred_xgb)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
precision, recall, _ = precision_recall_curve(y_test, y_pred_xgb)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#%%
le = LabelEncoder()
sparcs_2014['APR_RISK_OF_MORTALITY']=sparcs_2014['APR_RISK_OF_MORTALITY'].replace("NA","Minor")
le.fit(["Minor", "Moderate", "Major", "Extreme"])
y=le.transform(sparcs_2014['APR_RISK_OF_MORTALITY'])


equiv_mort = {'Minor':0, 'Moderate':1, 'Major':0, 'Extreme': 0}
sparcs_2014['y_binary']=sparcs_2014['APR_RISK_OF_MORTALITY'].map(equiv_mort)
y_binary=sparcs_2014['y_binary'].values

# One hot encoded
one_hot_encoded_X = pd.get_dummies(sparcs_2014[my_cols])
print("# of columns after one-hot encoding: {0}"\
      .format(len(one_hot_encoded_X.columns)))
one_hot_encoded_X=clean(one_hot_encoded_X)

y_pred_xgb = clf.predict_proba(one_hot_encoded_X)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_binary, y_pred_xgb)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

average_precision = average_precision_score(y_binary, y_pred_xgb)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
precision, recall, _ = precision_recall_curve(y_binary, y_pred_xgb)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


#%%