
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
import time


os.getcwd()
os.chdir('Documents\Analytics\Jobathon May 2021')

train=pd.read_csv('Data/train_s3TEQDk.csv')
test=pd.read_csv('Data/test_mSzZ8RL.csv')

traindf=pd.concat([train,test]).reset_index(drop=True)
traindf['Credit_Product'].fillna('Missing',inplace=True)


traindf.Age=traindf.Age.astype(str)
traindf.Vintage=traindf.Vintage.astype(str)

cat_cols=list(traindf.columns[traindf.dtypes=='object'].drop(['ID']))
num_cols=list(traindf.columns[traindf.dtypes!='object'].drop(['Is_Lead']))


params_cb={
    'cat_features': cat_cols,
    'random_seed': 123,
    'n_estimators': 3000,
    'colsample_bylevel': 0.3278135597822217,
    'depth': 7,
    'l2_leaf_reg':2,
    'learning_rate':0.0657241365564295}

traindf[cat_cols]=traindf[cat_cols].astype(str)
train_X=traindf[~traindf['Is_Lead'].isnull()]
train_X=train_X[num_cols+cat_cols+['Is_Lead']]

test_X=traindf[traindf['Is_Lead'].isnull()]
test_X=test_X[num_cols+cat_cols]


fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
cb_scores=[]
pred_cb=[]
imp_scores=[]
t1=time.time()
for idxT, idxV in fold.split(train_X,train_X.Is_Lead):
    X_train, X_test = train_X.iloc[idxT].drop('Is_Lead',1), train_X.iloc[idxV].drop('Is_Lead',1)
    y_train, y_test = train_X.iloc[idxT].Is_Lead, train_X.iloc[idxV].Is_Lead
    
    cb=CatBoostClassifier(early_stopping_rounds=50,eval_metric='AUC',**params_cb)
    cb.fit(X_train, y_train,eval_set=(X_test,y_test),plot=False, verbose=200)
    cb_scores.append(cb.get_best_score().get('validation').get('AUC'))
    test_X[cat_cols]=test_X[cat_cols].astype(str)
    pred_cb.append(cb.predict_proba(test_X)[:,1])
    imp_scores.append(cb.get_feature_importance())
    print ('The Local CV till now is {} for {} rounds and took {} minutes'.format(np.mean(cb_scores),len(cb_scores),(time.time()-t1)/60))


weights=cb_scores/np.sum(cb_scores)
print ('The Local CV is {}'.format(np.sum(weights*cb_scores)))

pred=np.sum(np.multiply(np.transpose(np.array(pred_cb)),weights),1)
submit=pd.DataFrame({'ID':test.ID,'Is_Lead':pred})
submit.to_csv('Submissions/submit17.csv',index=False)