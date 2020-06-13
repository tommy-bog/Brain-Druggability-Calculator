from collections import OrderedDict
import json
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw import IPythonConsole
import sklearn
import sklearn.model_selection
from sklearn.feature_selection import mutual_info_regression 
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
sc = StandardScaler()
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, PandasTools
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing
from rdkit.Chem.Draw import IPythonConsole
from collections import OrderedDict
import inspect
from sklearn.decomposition import PCA, KernelPCA

import numpy as np
import pandas as pd
import sklearn 
#import statsmodels.formula.api as sm
from sklearn.feature_selection import mutual_info_regression 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import statistics as st
import math

#==========================================================================

class RdkitDescriptors():
    def __init__(self, PATH, target_col, smiles_col):

        self.df = pd.read_csv(PATH, delimiter = ',')
        print(self.df.head())
        self.target_col = target_col
        self.smiles_col = smiles_col
        data = {'smiles': self.df[smiles_col], 'target': self.df[target_col]}
        self.df = pd.DataFrame(data)
        print(self.df.head())
        self.df['mol'] = self.df.apply(lambda row: Chem.MolFromSmiles(row['smiles']), axis = 1)
        print(self.df.head())

    def basic_rdkit(self):
        """
        Returns a dataframe with the ~200 cheap conformer independent
        descriptors availbale from RDKit
        """
        props_to_use = list()
        calc_props = OrderedDict(inspect.getmembers(Descriptors, inspect.isfunction))
        print(calc_props.keys())
        for key in list(calc_props.keys()):
            if key.startswith('_'):
                del calc_props[key]
        print('Found {} molecular descriptors in RDKIT'.format(len(calc_props)))
        for key,val in calc_props.items():
            self.df[key] = self.df['mol'].apply(val)
            if not True in np.isnan(self.df[key].tolist()):
                props_to_use.append(key)
        print("DF shape: ", self.df.shape)
        print(self.df.head())

        return self.df

    def rdkit_fingerprints(self):
        """
        Returns a dataframe including the RDKit molecular fingerprints
        """
        self.df['rdkitfp'] = self.df.apply(lambda row: Chem.RDKFingerprint(row['mol']), axis = 1)
        x_fp = self.df['rdkitfp'].values
        print(x_fp.shape)
        return self.df
        
#==========================================================================
     

rdkitdesc = RdkitDescriptors('Data.csv', 'LogKp', 'SMILES')
df = rdkitdesc.rdkit_fingerprints()

fps = np.empty((688, 2048))
for _, mol in enumerate(df['mol'].values):
    fp = Chem.RDKFingerprint(mol)
    fps[_] = fp
    
fp = np.concatenate((df, fps), axis = 1)
fp = pd.DataFrame(fp)

names = pd.read_csv('Data.csv')
names = names.iloc[:, 0]
names = pd.DataFrame(names)
df = np.concatenate((names, fp), axis = 1)
df = pd.DataFrame(df)



#==========================================================================


dataset = df
X_data = dataset.iloc[:, 5:2060]
X = dataset.iloc[:, 5:2060].values
# X = sc.fit_transform(X)
y = dataset.iloc[:, 2].values
y = sc.fit_transform(dataset[2].values.reshape(-1, 1))

descriptors = X_data.columns


fis=np.empty((10, 2048))
for i in range(10):
      model =  GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, random_state=i)
      model.fit(X, y)     
      fi = model.feature_importances_
      fis[i] = fi
     
avg_fi    = np.mean(fis, axis = 0)
sorted_fi, sorted_ix = np.sort(avg_fi), np.argsort(avg_fi)


##########################################################################################################################
'''
# save best features/bits

good_f, good_ix    = sorted_fi[-200:], sorted_ix[-200:]
good_desc = [descriptors[ix] for ix in good_ix]

import pickle

filename1 = 'good_f.sav'
pickle.dump(good_f, open(filename1, 'wb'))
filename2 = 'good_ix.sav'
pickle.dump(good_ix, open(filename2, 'wb'))

'''
##########################################################################################################################


# load best features/bits

filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))


'''
##########################################################################################################################

avg_fi    = np.mean(fis, axis = 0)
sorted_fi, sorted_ix = np.sort(avg_fi), np.argsort(avg_fi)
good_f, good_ix    = sorted_fi[-120:], sorted_ix[-120:]
good_desc = [descriptors[ix] for ix in good_ix]


X = X[:, good_ix]

##########################################################################################################################


# MLR

from sklearn.linear_model import LinearRegression
import math


# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=i)
      model =  LinearRegression()
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# Best split is random_state = 83

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=83)



model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

from sklearn.model_selection import cross_val_score
import statistics as st

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=10, scoring = 'neg_root_mean_squared_error')

avgR2_10_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))




##########################################################################################################################

# RFR

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(200, random_state=1)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=10, scoring = 'neg_root_mean_squared_error')

avgR2_10_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))


'''
##########################################################################################################################

# SVR

from sklearn.svm import SVR


# determining best number of bits
r_squared_tr_vals = np.empty((160, 1))
r_squared_te_vals = np.empty((160, 1))
r_squared_CV_vals = np.empty((160, 1))
for i in range(1,160):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=20)
      model = SVR(kernel = 'rbf', C=8, epsilon = 0.01)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV


filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-107:]
X = dataset.iloc[:, 5:2060].values
X = X[:, good_ix]


#================================================================================


# Adding transporter info: 
 # pgp
filename = 'pgp_model.sav'
pgp_model = pickle.load(open(filename, 'rb'))

filename = 'pgp_ix.sav'
pgp_ix = pickle.load(open(filename, 'rb'))

X = dataset.iloc[:, 5:2060].values
X = X[:, pgp_ix]

trans_pgp_raw = pgp_model.predict(X)

filename = 'pgp_y.sav'
pgp_y = pickle.load(open(filename, 'rb'))

pgp_y_tr = sc.fit_transform(pgp_y.reshape(-1, 1))

trans_pgp = sc.inverse_transform(trans_pgp_raw)
trans_pgp = pd.DataFrame(trans_pgp)


# Adding transporter info: 
 # bcrp
filename = 'bcrp_model.sav'
bcrp_model = pickle.load(open(filename, 'rb'))

filename = 'bcrp_ix.sav'
bcrp_ix = pickle.load(open(filename, 'rb'))

X = dataset.iloc[:, 5:2060].values
X = X[:, bcrp_ix]

trans_bcrp_raw = bcrp_model.predict(X)

filename = 'bcrp_y.sav'
bcrp_y = pickle.load(open(filename, 'rb'))

bcrp_y_tr = sc.fit_transform(bcrp_y.reshape(-1, 1))

trans_bcrp = sc.inverse_transform(trans_bcrp_raw)
trans_bcrp = pd.DataFrame(trans_bcrp)


# Adding transporter info: 
 # oct1
filename = 'oct1_model.sav'
oct1_model = pickle.load(open(filename, 'rb'))

filename = 'oct1_ix.sav'
oct1_ix = pickle.load(open(filename, 'rb'))

X = dataset.iloc[:, 5:2060].values
X = X[:, oct1_ix]

trans_oct1_raw = oct1_model.predict(X)

filename = 'oct1_y.sav'
oct1_y = pickle.load(open(filename, 'rb'))

oct1_y_tr = sc.fit_transform(oct1_y.reshape(-1, 1))

trans_oct1 = sc.inverse_transform(trans_oct1_raw)
trans_oct1 = pd.DataFrame(trans_oct1)


# Adding transporter info: 
 # oct2
filename = 'oct2_model.sav'
oct2_model = pickle.load(open(filename, 'rb'))

filename = 'oct2_ix.sav'
oct2_ix = pickle.load(open(filename, 'rb'))

X = dataset.iloc[:, 5:2060].values
X = X[:, oct2_ix]

trans_oct2_raw = oct2_model.predict(X)

filename = 'oct2_y.sav'
oct2_y = pickle.load(open(filename, 'rb'))

oct2_y_tr = sc.fit_transform(oct2_y.reshape(-1, 1))

trans_oct2 = sc.inverse_transform(trans_oct2_raw)
trans_oct2 = pd.DataFrame(trans_oct2)



#================================================================================

filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-107:]
X = dataset.iloc[:, 5:2060].values
X = X[:, good_ix]


X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)

#================================================================================



# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
r_squared_CV_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=i)
      model = SVR(kernel = 'rbf', C=5, epsilon = 0.01)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# Best split is random_state = 62 , 73 x29x


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=29)


# determining best number of bits
r_squared_tr_vals = np.empty((200, 1))
r_squared_te_vals = np.empty((200, 1))
r_squared_CV_vals = np.empty((200, 1))
for i in range(1,200):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=29)
      model = SVR(kernel = 'rbf', C=8, epsilon = 0.01)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV



filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-145:]

filename = 'kp_ix.sav'
pickle.dump(good_ix, open(filename, 'wb'))

X = dataset.iloc[:, 5:2060].values
X = X[:, good_ix]

X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=29)





model = SVR(kernel = 'rbf', C=3, epsilon = 0.0001)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=10, scoring = 'neg_root_mean_squared_error')

avgR2_10_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))




filename = 'kp_model.sav'
pickle.dump(model, open(filename, 'wb'))


filename = 'kp_y.sav'
y = dataset.iloc[:, 2].values
pickle.dump(y, open(filename, 'wb'))




model = GradientBoostingRegressor()
model.fit(X,y)

fi = model.feature_importances_






##########################################################################################################################

# GBR 


# determining best number of bits
r_squared_tr_vals = np.empty((200, 1))
r_squared_te_vals = np.empty((200, 1))
r_squared_CV_vals = np.empty((200, 1))
for i in range(100,200):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=83)
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=600, random_state=0)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV


filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-176:]
X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)
X = X[:, good_ix]

X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)


# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
r_squared_CV_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=i)
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=600, random_state=0)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# Best split is random_state = 15, 6, 83 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=15)


# Determining max features
r_squared_tr_vals = np.empty((181, 1))
r_squared_te_vals = np.empty((181, 1))
r_squared_CV_vals = np.empty((181, 1))
for i in range(1,181):
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=300, random_state=0, max_features=i)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# Optimal max features (15) = 21, 18, 6, 7
# Optimal max features (6) = 48, 116, 136

# Determining random_state
r_squared_tr_vals = np.empty((30, 1))
r_squared_te_vals = np.empty((30, 1))
r_squared_CV_vals = np.empty((30, 1))
for i in range(0,30):
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=300, random_state=i, max_features=116)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# (15) Optimal random_state (21) = 27, 0
# (15) Optimal random_state (18) = 0
# (15) Optimal random_state (6) = 12, 4
# (15) Optimal random_state (7) = 2, 29
# (6) Optimal random_state (48) = 0
# (6) Optimal random_state (116) = 10

# Optimize learning rate
lr=np.empty((200, 1))
r_squared_tr_vals = np.empty((200, 1))
r_squared_te_vals = np.empty((200, 1))
r_squared_CV_vals = np.empty((200, 1))
for i in range(1, 200):
      model =  GradientBoostingRegressor(learning_rate=i*0.001, n_estimators=300, random_state=10, max_features=116)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      lr[i]=i*0.001
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV

# best lr (spl 15) (mf 21) (rs 27) = 0.01
# best lr (spl 15) (mf 18) (rs 0) = 0.107
# best lr (spl 15) (mf 6) (rs 12)  = 0.098
# best lr (spl 15) (mf 6) (rs 4)  = 0.103
# best lr (spl 15) (mf 7) (rs 2)  = 0.102
# best lr (spl 15) (mf 7) (rs 29)  = 0.132
# best lr (spl 6) (mf 48) (rs 0)  = 0.078
# best lr (spl 6) (mf 116) (rs 10)  = 0.1


# spl15 21mf =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=600, random_state=27, max_features=21)
# splt15 18mf model =  GradientBoostingRegressor(learning_rate=0.107, n_estimators=400, random_state=0, max_features=18)
# spl15 *82,65* model =  GradientBoostingRegressor(learning_rate=0.098, n_estimators=800, random_state=12, max_features=6)
# splt15 *79,67* model =  GradientBoostingRegressor(learning_rate=0.103, n_estimators=600, random_state=4, max_features=6)

# splt15 *83,65* model =  GradientBoostingRegressor(learning_rate=0.102, n_estimators=520, random_state=2, max_features=7)

# spl6 *83,64* model =  GradientBoostingRegressor(learning_rate=0.078, n_estimators=320, random_state=0, max_features=48)

# best model at split 15: 
      
model =  GradientBoostingRegressor(learning_rate=0.102, n_estimators=520, random_state=2, max_features=7)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=10, scoring = 'neg_root_mean_squared_error')

avgR2_10_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))



y_pred_train_values = sc.inverse_transform(y_pred_train)
y_train_values = sc.inverse_transform(y_train)

y_pred_test_values = sc.inverse_transform(y_pred_test)
y_test_values = sc.inverse_transform(y_test)



sns.distplot(dataset[2].values.reshape(-1, 1))



filename = 'Kp_model.sav'
pickle.dump(model, open(filename, 'wb'))

Kp_model = pickle.load(open(filename, 'rb'))

y_pred_train = Kp_model.predict(X_train)




##########################################################################################################################

# BGP Model


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF



# determining best number of bits
r_squared_tr_vals = np.empty((200, 1))
r_squared_te_vals = np.empty((200, 1))
r_squared_CV_vals = np.empty((200, 1))
for i in range(0,200):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=3)
      model = GaussianProcessRegressor(kernel=RBF(5), random_state=1, alpha=1e-8, n_restarts_optimizer=4, normalize_y=True).fit(X_train, y_train)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV


filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-80:]
X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)
X = X[:, good_ix]

X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)



# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
r_squared_CV_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=i)
      model = GaussianProcessRegressor(kernel=RBF(5), random_state=1, alpha=1e-8, n_restarts_optimizer=4, normalize_y=True).fit(X_train, y_train)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# Best split is random_state = 3


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=3)

model = GaussianProcessRegressor(kernel=RBF(5), random_state=1, alpha=1e-8, n_restarts_optimizer=4, normalize_y=True).fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=10, scoring = 'neg_root_mean_squared_error')

avgR2_10_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))



plt.plot(X_test, y_test, 'kx', mew=2)
line, = plt.plot(X_test, mean, lw=2)
_ = plt.fill_between(X_test[:,0], mean[:,0] - 2*np.sqrt(y_std_test[:,0]), mean[:,0] + 2*np.sqrt(y_std_test[:,0]), color=line.get_color(), alpha=0.2)



######################################################################################################################################

# BDT

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# determining best number of bits
r_squared_tr_vals = np.empty((155, 1))
r_squared_te_vals = np.empty((155, 1))
for i in range(1,155):
      good_f, good_ix    = sorted_fi[-i:], sorted_ix[-i:]
      good_desc = [descriptors[ix] for ix in good_ix]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=73)
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50, learning_rate=0.8, random_state=1)
      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)

# 
good_f, good_ix    = sorted_fi[-95:], sorted_ix[-95:]
good_desc = [descriptors[ix] for ix in good_ix]
X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)
X = X[:, good_ix]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=73)


# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=i)
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50, learning_rate=0.8, random_state=1)
      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# = 83

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=83)


# Determining estimators
r_squared_tr_vals = np.empty((130, 1))
r_squared_te_vals = np.empty((130, 1))
for i in range(1,130):
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=i, learning_rate=0.8, random_state=1)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# Optimal estimators = 125


'''
# Determining random_state
r_squared_tr_vals = np.empty((30, 1))
r_squared_te_vals = np.empty((30, 1))
for i in range(0,30):
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=81, learning_rate=0.8, random_state=i)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# Optimal random_state = 0
'''


# Optimize learning rate
lr=np.empty((200, 1))
r_squared_tr_vals = np.empty((200, 1))
r_squared_te_vals = np.empty((200, 1))
for i in range(1, 200):
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=125, learning_rate=i*0.01, random_state=1)
      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      lr[i]=i*0.01
# best lr = 1.2



model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=125, learning_rate=1.2, random_state=1)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=10, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=10, scoring = 'neg_root_mean_squared_error')

avgR2_10_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))






######################################################################################################################################


# Visualising the TR results
plt.scatter(y_train, y_pred_train, color = 'red')
plt.title('BCRP pIC50')
plt.xlabel('Actuals')
plt.ylabel('Predictions')
plt.show()

# Visualising the model on TR
X_train_drug = list(range(1,26))
plt.scatter(X_train_drug, y_train, color = 'red')
plt.plot(X_train_drug, y_pred_train, color = 'blue')
plt.title('BCRP pIC50')
plt.xlabel('Drug Number')
plt.ylabel('Pgp LogER')
plt.show()

# Visualising the TE results
plt.scatter(y_test, y_pred_test, color = 'blue')
plt.title('BCRP pIC50')
plt.xlabel('Actuals')
plt.ylabel('Predictions')
_ = plt.plot([-1, 2], [-1, 2])
plt.show()

# Visualising the model on TE 
X_test_drug = list(range(1,8))
plt.scatter(X_test_drug, y_test, color = 'red')
plt.plot(X_test_drug, y_pred_test, color = 'blue')
plt.title('BCRP pIC50')
plt.xlabel('Drug Number')
plt.ylabel('Pgp LogER')
plt.show()




saved_pgp_model = pickle.dumps(model) 

pgp_model = pickle.loads(saved_pgp_model) 
  
y_pred_train = pgp_model.predict(X_train)
y_pred_train_values = sc.inverse_transform(y_pred_train)




















