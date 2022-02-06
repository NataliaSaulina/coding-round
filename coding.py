import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold,KFold, train_test_split, cross_val_predict
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization

#Load data
df=pd.read_csv('coding_round_data.csv')
#Check for missing values in each column
for name in df.columns:
    if df[name].isnull().values.any():
        print('Column %s contains NaN'% name)

#Create target
df['target'] = df['Revenue'].astype(int)
Y = df['target'].values

#Convert columns that contain strings to numbers to be able to compute mutual information with target
month_to_num={'Jan':1,'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'June':6, 'Jul':7, 'Aug':8,'Sep':9, 'Oct':10, 'Nov':11,
              'Dec':12}
df['Month']=df["Month"].map(month_to_num)
oe=OrdinalEncoder()
df['VisitorType']=oe.fit_transform(df['VisitorType'].values.reshape(-1,1))
df['Weekend']=df['Weekend'].astype(int)

#Compute mutualinformation between each categorical column and target
cat_cols=['Month','OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']


cnt_cat_cols=len(cat_cols)

cat_scores={}
for name in cat_cols:
    cat_scores[name]=mutual_info_classif(df[name].values.reshape(-1,1),Y,discrete_features=True)[0]
cat_ordered_by_relevance=sorted(cat_scores.items(),key= lambda x: x[1],reverse=True)

# Count number of unique values in each categorical column.
# If more than 100, we need to think if Embedding is better that OneHotEncoding
to_check=[]
distinct={} # this dictionary is used when we do encoding
for name in cat_cols:
    cur=len(df[name].unique())
    distinct[name]=cur
    if cur>100:
        to_check.append(name)
if len(to_check):
    print('The following categorical columns have >100 categories')
    print(to_check)



#Create features and binary target column
X=df.drop(['Revenue','target'],axis=1)


#Check for correlations in numerical columns
tmp=X.iloc[:,:10]
cor_mat=np.corrcoef(tmp,rowvar=False)
high_cor_cols=[]
for i in range(10):
    for j in range(i+1,10):
        if cor_mat[i][j]>0.6:
            high_cor_cols.append((i,j))
print('The following pairs of columns are highly correlated')
print(high_cor_cols)

#There is high correlation between 0 and 1, 2 and 3, 4 and 5, 6 and 7 pairs of columns
# We keep only 0,2,4 and 7 in feature matrix

X=X.drop(['Administrative_Duration',
       'Informational_Duration', 'ProductRelated_Duration','BounceRates'],axis=1)





#List columns kept in feature matrix

num_cols=['Administrative','Informational','ProductRelated','ExitRates', 'PageValues', 'SpecialDay']
cnt_num_cols=len(num_cols)


#Check ouliets in numerical features
X_num=X[num_cols]
q_99 =X_num.quantile(0.99)


#Let us drop outliers. Alternatively we could have gaussianized numerical columns.

for i in range(len(num_cols)):
    if i==0:
        mask= X_num.iloc[:,i] > q_99[i]
    else:
        mask = np.logical_or(mask,X_num.iloc[:,i] > q_99[i])

X_=X[~mask]
Y_=Y[~mask]


corr_num_feat_w_target={}
for name in num_cols:
    cur=np.corrcoef(X_[name],Y_)
    corr_num_feat_w_target[name]=cur[0][1]

num_ordered_by_relevance=sorted(corr_num_feat_w_target.items(),key= lambda x: abs(x[1]),reverse=True)


###### Part1. Build Logistic Regression model


def train_and_test_LR(x,y,keep_cat,keep_num,do_grid_search=False):

    #x - DataFrame of features
    # y - target (1D numpy array)
    # keep_cat - list of cat column names
    # keep_num - list of num column names
    #do_grid_search - Bool

    # This function trains Logistic Regression. It retains dictionary that contains
    # validation accuracy and test accuracy,
    # beta-coefficients
    # intercept
    # if do_grid_search = True, dictionary also contains best hyperparameters.


    info={}

    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2, random_state=7)
    ohe = OneHotEncoder(handle_unknown='ignore')
    ss= StandardScaler()
    if len(keep_cat)>0: # I always keep at least 1 num feature
        pipe=ColumnTransformer([('num',ss,keep_num),('cat',ohe,keep_cat)])
    else:
        pipe = ColumnTransformer([('num', ss, keep_num)])

    train_data=pipe.fit_transform(X_train)
    test_data=pipe.transform(X_test)


    if do_grid_search:

        model=LogisticRegression()
        solver=['saga']
        penalty=['l1']
        c_values=[0.01,0.001,0.0005,0.0003,0.0001]

        class_weight=['balanced','none']
        grid=dict(solver=solver,penalty=penalty,C=c_values, class_weight=class_weight)

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search=GridSearchCV(estimator=model,param_grid=grid, cv=cv, scoring='accuracy',n_jobs=-1)

        grid_result=grid_search.fit(train_data,Y_train)
        info['best_val_accuracy']=grid_result.best_score_
        info['best_hyperparams']=grid_result.best_params_



        clf=grid_search.best_estimator_

        clf.fit(train_data,Y_train)
        info['beta-coef']=clf.coef_
        info['intercept']=clf.intercept_


        Y_test_pred=clf.predict(test_data)


        conf_mat_test=confusion_matrix(Y_test,Y_test_pred)
        test_acc=(conf_mat_test[0][0]+conf_mat_test[1][1])/len(Y_test)

        info['test_accuracy']=test_acc

        return info
    else:
        clf=LogisticRegression(penalty='none',class_weight='balanced',solver='saga',verbose=0,max_iter=100)
        Y_train_pred=cross_val_predict(clf,train_data,Y_train,n_jobs=-1, cv=5,verbose=0)


        conf_mat_train = confusion_matrix(Y_train, Y_train_pred)
        val_acc = (conf_mat_train[0][0] + conf_mat_train[1][1]) / len(Y_train)
        info['val_accuracy']=val_acc

        clf.fit(train_data, Y_train)
        info['beta-coef'] = clf.coef_
        info['intercept'] = clf.intercept_

        Y_test_pred = clf.predict(test_data)

        conf_mat_test = confusion_matrix(Y_test, Y_test_pred)
        test_acc = (conf_mat_test[0][0] + conf_mat_test[1][1]) / len(Y_test)

        info['test_accuracy'] = test_acc
        return info

# LASSO (Logistic Regression with 'l1' penalty) with grid search  allows to select features
res={}
res['l1']=train_and_test_LR(X_,Y_,cat_cols,num_cols,do_grid_search=True)

print('LASSO achieves the best accuracy in a model with a single feature "PageValues"')

#Now I try to check this by training LR without penalty on a data with more features added
# I used ordered (by relevance to Y) lists of features constructed previously
#num_ordered_by_relevance  #cat_ordered_by_relevance  to identify candidate features

keep_cat_=['TrafficType','Month']
keep_num_=['PageValues','ExitRates','ProductRelated']

for i in range(1,3): # I always keep at least 1 num feature
    for j in range(2):
        keep_cat=keep_cat_[:j]
        keep_num=keep_num_[:i]
        keep=keep_cat + keep_num
        res[(i,j)]=train_and_test_LR(X_[keep],Y_,keep_cat,keep_num)



#Inspecting res, I see that model with a single feature 'PageValues' indeed has the best validation accuracy.
#The next best is model (1,1) with 'PageValues' and 'TrafficType'. Model (1,1) gives the best accuracy on test data


#I now try to improve both validation and test accuracy by buiding Neural Network model


X_cat=X_[cat_cols]
X_num=X_[num_cols]



ohe=OneHotEncoder()
X_cat_enc=ohe.fit_transform(X_cat)

X_cat_enc_np=X_cat_enc.toarray()
X_use= np.hstack((X_num, X_cat_enc_np))



X_train,X_test,Y_train,Y_test=train_test_split(X_use,Y_,test_size=0.2, random_state=7)



ss=StandardScaler()
train_num =ss.fit_transform(X_train[:,:cnt_num_cols])
test_num=ss.transform(X_test[:,:cnt_num_cols])

train_data=np.hstack((train_num, X_train[:,cnt_num_cols:]))
test_data=np.hstack((test_num, X_test[:,cnt_num_cols:]))



def build_model(hp):
    input_dim=train_data.shape[1]
    hp_units = hp.Int('units', 4, 16, step=4)

    model=Sequential([Dense(hp_units,input_dim= input_dim, activation ='elu'),
    BatchNormalization(),
    Dropout(hp.Float('dropout',0,0.5,step=0.1)),
    Dense(1,activation='sigmoid')])

    opt= keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-4,1e-3,1e-2]))
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model



tuner_bo= BayesianOptimization(build_model,objective='val_accuracy',max_trials=30)
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)



tuner_bo.search(train_data,Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=0)

best_model=tuner_bo.get_best_models(1)[0]

best_hp=tuner_bo.get_best_hyperparameters(1)[0]
print('NN  performs best with  units, dropout, learning_rate given by')
print(best_hp.get('units'),best_hp.get('dropout'),best_hp.get('learning_rate'))

model=tuner_bo.hypermodel.build(best_hp)
train_res=model.fit(train_data,Y_train, validation_split=0.2,  epochs=50, callbacks=[stop_early],verbose=0)

val_acc_per_epoch = train_res.history['val_accuracy']
best_val_acc=max(val_acc_per_epoch)
best_epoch = val_acc_per_epoch.index(best_val_acc) + 1
print('Best epoch: %d' % (best_epoch,))
print('NN  validation accuracy %.2f'% (100*best_val_acc))

#Retrain with epochs = best_epoch to obtain the model with best validation accuracy
model=tuner_bo.hypermodel.build(best_hp)
train_res=model.fit(train_data,Y_train, validation_split=0.2,  epochs=best_epoch,verbose=0)

# Evaluate on test data
eval_res=model.evaluate(test_data,Y_test)
print('NN evaluation on test data gives accuracy %.2f' %(eval_res[1]*100))
pdb.set_trace()