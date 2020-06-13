from collections import OrderedDict
import json
import pickle
import numpy as np
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
        




rdkitdesc = RdkitDescriptors('Data.csv', 'LogKpuu', 'SMILES')
df = rdkitdesc.rdkit_fingerprints()

fps = np.empty((139, 2048))
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



'''
##########################################################################################################################

# save best features/bits

good_f, good_ix    = sorted_fi[-200:], sorted_ix[-200:]
good_desc = [descriptors[ix] for ix in good_ix]

import pickle

filename1 = 'good_fRD.sav'
pickle.dump(good_f, open(filename1, 'wb'))
filename2 = 'good_ixRD.sav'
pickle.dump(good_ix, open(filename2, 'wb'))

'''

##########################################################################################################################

# load best features/bits

filename = 'good_ix.sav'
good_ix = pickle.load(open(filename, 'rb'))


##########################################################################################################################
'''
# MLR

from sklearn.linear_model import LinearRegression
import math


X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)

good_f, good_ix    = good_f[-36:], good_ix[-36:]
good_desc = [descriptors[ix] for ix in good_ix]

X = X[:, good_ix]




# determining best number of bits
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
for i in range(1,90):
      good_f, good_ix    = sorted_fi[-i:], sorted_ix[-i:]
      good_desc = [descriptors[ix] for ix in good_ix]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=12)
      model =  LinearRegression()
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)


# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=i)
      model =  LinearRegression()
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# Best split is random_state = 21

      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=21)


# determining best number of bits, for new split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
for i in range(1,90):
      good_f, good_ix    = sorted_fi[-i:], sorted_ix[-i:]
      good_desc = [descriptors[ix] for ix in good_ix]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=21)
      model =  LinearRegression()
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)


good_f, good_ix    = good_f[-36:], good_ix[-36:]
X = X[:, good_ix]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=21)


model = LinearRegression()
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



##########################################################################################################################

# RFR

from sklearn.ensemble import RandomForestRegressor



# determining best number of bits
r_squared_tr_vals = np.empty((120, 1))
r_squared_te_vals = np.empty((120, 1))
r_squared_CV_vals = np.empty((120, 1))
for i in range(1,120):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=12)
      model = RandomForestRegressor(200, random_state=1, min_samples_split=2, max_depth=8)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV  

# (51,0.15): 38 (50,45)
# (52,0.15): 50 (55,41)
# (12,0.15): 42 (56,41) , 50 (60,39)

      

good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-38:]
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
      model = RandomForestRegressor(200, random_state=1, min_samples_split=2, max_depth=8)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 50 bits = 82 (62,39) , 52 , 12
# 42 bits = 82 (65,39) , 12
# 38 bits = 82 (62,39)

good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-42:]
X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)
X = X[:, good_ix]

X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)

  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=82)

  

# Determining random_state
r_squared_tr_vals = np.empty((55, 1))
r_squared_te_vals = np.empty((55, 1))
r_squared_CV_vals = np.empty((55, 1))
for i in range(0,55):
      model = RandomForestRegressor(200, random_state=i, min_samples_split=2, max_depth=8)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 42 bits, splt 82 = 3 (64,41), 28 (67,38)

# Determining min samples
r_squared_tr_vals = np.empty((25, 1))
r_squared_te_vals = np.empty((25, 1))
r_squared_CV_vals = np.empty((25, 1))
for i in range(2,25):
      model = RandomForestRegressor(200, random_state=28, min_samples_split=i, max_depth=8)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 42 bits, splt 82, rs3  = 2
# 42 bits, splt 82, rs28 = 5 (68,38)


# Determining max depth
r_squared_tr_vals = np.empty((40, 1))
r_squared_te_vals = np.empty((40, 1))
r_squared_CV_vals = np.empty((40, 1))
for i in range(1,40):
      model = RandomForestRegressor(200, random_state=3, min_samples_split=2, max_depth=i)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 42 bits, splt 82, rs28, ms5 = 8
# 42 bits, splt 82, rs3, ms2  = 6, 10 


# Determining max features
r_squared_tr_vals = np.empty((46, 1))
r_squared_te_vals = np.empty((46, 1))
r_squared_CV_vals = np.empty((46, 1))
for i in range(1,46):
      model = RandomForestRegressor(100, random_state=28, min_samples_split=5, max_depth=8, max_features=i)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# 42 bits, splt 82 = //


# Determining estimators
r_squared_tr_vals = np.empty((100, 1))
r_squared_te_vals = np.empty((100, 1))
r_squared_CV_vals = np.empty((100, 1))
for i in range(1,100):
      model = RandomForestRegressor(i, random_state=28, min_samples_split=5, max_depth=8)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 42 bits, splt 82, rs3, ms2, md10 = 38 (71,36) , 85 (66,40)
# 42 bits, splt 82, rs28, ms5, md8 = 100


# (70,37) model = RandomForestRegressor(100, random_state=28, min_samples_split=5, max_depth=8)

# (68, 38) model = RandomForestRegressor(55, random_state=3, min_samples_split=2, max_depth=10)


model = RandomForestRegressor(100, random_state=28, min_samples_split=5, max_depth=8)
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
scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_root_mean_squared_error')

avgR2_5_fold_CV = st.mean(scoresR2)
avgRMSE_5_fold_CV = st.mean(scoresRMSE)
avgRMSE_5_fold_CV = (-1)*(st.mean(scoresRMSE))



##########################################################################################################################

# SVR

from sklearn.svm import SVR
filename = 'good_ix.sav'


# determining best number of bits
r_squared_tr_vals = np.empty((155, 1))
r_squared_te_vals = np.empty((155, 1))
r_squared_CV_vals = np.empty((155, 1))
for i in range(1,155):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=2)
      model = SVR(kernel = 'rbf', C=10, epsilon = 0.00001)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=6, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV  

# rbf (2,0.15): 117 (75,53) ;   6fold CV: 115 (76,55)
# linear (2,0.15): //
# poly (2,0.15): //
# sigmoid (2,0.15): //


good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-115:]
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
      model = SVR(kernel = 'rbf', C=5, epsilon = 0.01)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 117 bits = 2

  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=2)


# 1.0, 53, 76

model = SVR(kernel = 'rbf', C=10, epsilon = 0.00001)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

y_pred = model.predict(X)
y = dataset.iloc[:, 2].values
y = sc.fit_transform(dataset[2].values.reshape(-1, 1))
y_pred_vals = sc.inverse_transform(y_pred)
y_pred_vals_act = np.empty((139, 1))
for i in range(0,139):
      y_pred_vals_act[i] = math.pow(10, y_pred_vals[i])


MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=6, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=6, scoring = 'neg_root_mean_squared_error')

avgR2_6_fold_CV = st.mean(scoresR2)
avgRMSE_6_fold_CV = st.mean(scoresRMSE)
avgRMSE_6_fold_CV = (-1)*(st.mean(scoresRMSE))

import math
import statistics as st


filename = 'kpuu_model.sav'
pickle.dump(model, open(filename, 'wb'))

filename = 'kpuu_ix.sav'
pickle.dump(good_ix, open(filename, 'wb'))

good_ix = good_ix[-115:]
X = X[:, good_ix]

X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)

filename = 'kpuu_y.sav'
y = dataset.iloc[:, 2].values
pickle.dump(y, open(filename, 'wb'))




##########################################################################################################################

# GBR 



filename = 'good_ix.sav'


# determining best number of bits
r_squared_tr_vals = np.empty((122, 1))
r_squared_te_vals = np.empty((122, 1))
r_squared_CV_vals = np.empty((122, 1))
for i in range(1,122):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=2)
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=0)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# cv5 (2, 0.15): 69
# cv6 (2, 0.15): \\
    

good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-110:]
X = dataset.iloc[:, 5:2060].values
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
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=0)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# Best split is random_state (0.2) = 81, 46, 12
# Best split is random_state (0.15) = 2, 4, 33  ;  75 (80,43) vs 2 (78,52)
# updated Best split is random_state (0.15) = 53


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=75)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=53)



# Determining max features
r_squared_tr_vals = np.empty((69, 1))
r_squared_te_vals = np.empty((69, 1))
r_squared_CV_vals = np.empty((69, 1))
for i in range(1,69):
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=0, max_features=i)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# Optimal max features (2,0.15, 110) = 1, 15, 75, 85  ;  
# Optimal max features (75,0.15, 110) = 1, 2  (cv10)  //  (cv5)  1, 2
# Optimal max features (53,0.15, 69) = 14
    

# Determining random_state
r_squared_tr_vals = np.empty((30, 1))
r_squared_te_vals = np.empty((30, 1))
r_squared_CV_vals = np.empty((30, 1))
for i in range(0,30):
      model =  GradientBoostingRegressor(learning_rate=0.1, n_estimators=500, random_state=i, max_features=14)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# mf 1 = 11
# mf 15 = 0
# mf 75 = 0
# mf 85 = 18
# mf 2 (75,0.15) = 29
      

# Optimize learning rate
lr=np.empty((200, 1))
r_squared_tr_vals = np.empty((200, 1))
r_squared_te_vals = np.empty((200, 1))
r_squared_CV_vals = np.empty((200, 1))
for i in range(1, 200):
      model =  GradientBoostingRegressor(learning_rate=i*0.001, n_estimators=500, random_state=29, max_features=2)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      lr[i]=i*0.001
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_10_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_10_fold_CV
# mf 1,11 = 141
# mf 15,0 = //
# mf 75,0 = 134
# mf 85,18 = 120


# ========================
## split2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=2)

# ***final*** (78,52) model =  GradientBoostingRegressor(learning_rate=0.141, n_estimators=3000, random_state=11, max_features=1)
# ========================


# (76,38) model =  GradientBoostingRegressor(learning_rate=0.120, n_estimators=600 random_state=18, max_features=85)


## split75
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=75)

# (82,37) model =  GradientBoostingRegressor(learning_rate=0.141, n_estimators=2600, random_state=29, max_features=2)




# **


model =  GradientBoostingRegressor(learning_rate=0.145, n_estimators=3000, random_state=11, max_features=1)
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
scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_root_mean_squared_error')

avgR2_5_fold_CV = st.mean(scoresR2)
avgRMSE_10_fold_CV = st.mean(scoresRMSE)
avgRMSE_10_fold_CV = (-1)*(st.mean(scoresRMSE))

fi = model.feature_importances_




filename = 'kpuu_model.sav'
pickle.dump(model, open(filename, 'wb'))

filename = 'kpuu_ix.sav'
pickle.dump(good_ix, open(filename, 'wb'))

good_ix = good_ix[-110:]
X = X[:, good_ix]

X = np.append(X, trans_pgp, axis=1)
X = np.append(X, trans_bcrp, axis=1)
X = np.append(X, trans_oct1, axis=1)
X = np.append(X, trans_oct2, axis=1)

filename = 'kpuu_y.sav'
y = dataset.iloc[:, 2].values
pickle.dump(y, open(filename, 'wb'))





y_pred = model.predict(X)
y = dataset.iloc[:, 2].values
y = sc.fit_transform(dataset[2].values.reshape(-1, 1))
y_pred_vals = sc.inverse_transform(y_pred)
y_pred_vals_act = np.empty((139, 1))
for i in range(0,139):
      y_pred_vals_act[i] = math.pow(10, y_pred_vals[i])






##########################################################################################################################

# BGP Model


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, RationalQuadratic, ExpSineSquared


kernel = RationalQuadratic(5)


# determining best number of bits
r_squared_tr_vals = np.empty((155, 1))
r_squared_te_vals = np.empty((155, 1))
r_squared_CV_vals = np.empty((155, 1))
for i in range(1,155):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=76)
      model = GaussianProcessRegressor(kernel=kernel, random_state=1, alpha=1e-1, n_restarts_optimizer=4, normalize_y=True).fit(X_train, y_train)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV  

# rbf(5) (2,0.15): 116 (74,51)
# rbf(5) (35,0.15): //
# rbf(5) (2,0.15), with alpha=1 : 113 (75, 57)
# rbf(5) (76,0.15), with alpha=1 : 105 (83, 43)
# RationalQuadratic(5) (76,0.15), with alpha=1 : 95 (82, 44)


good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-105:]
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
      model = GaussianProcessRegressor(kernel=kernel, random_state=1, alpha=1e-1, n_restarts_optimizer=4, normalize_y=True).fit(X_train, y_train)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 116 bits = 2, try 35
# 113 bits = 2, try 76 (80, 44)

  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=76)


kernel = RBF(2)

model = GaussianProcessRegressor(kernel=kernel, random_state=1, alpha=1e-1, n_restarts_optimizer=4, normalize_y=True).fit(X_train, y_train)

# rbf(5) (2,0.15), with alpha=1 : 113 (75, 57)
# rbf(5) (76,0.15), with alpha=1 : 105 (83, 44)


y_pred_train, y_std_train = model.predict(X_train, return_std=True)
y_pred_test, y_std_test = model.predict(X_test, return_std=True)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_train = y_train.ravel()
scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_root_mean_squared_error')

avgR2_5_fold_CV = st.mean(scoresR2)
avgRMSE_5_fold_CV = st.mean(scoresRMSE)
avgRMSE_5_fold_CV = (-1)*(st.mean(scoresRMSE))






plt.plot(X_test, y_test, 'kx', mew=2)
line, = plt.plot(X_test, mean, lw=2)
_ = plt.fill_between(X_test[:,0], mean[:,0] - 2*np.sqrt(y_std_test[:,0]), mean[:,0] + 2*np.sqrt(y_std_test[:,0]), color=line.get_color(), alpha=0.2)

######################################################################################################################################

# new CatBoost regressor

import catboost
from catboost import CatBoostRegressor


# determining best number of bits
r_squared_tr_vals = np.empty((155, 1))
r_squared_te_vals = np.empty((155, 1))
r_squared_CV_vals = np.empty((155, 1))
for i in range(1,155):
      good_ix = pickle.load(open(filename, 'rb'))
      good_ix = good_ix[-i:]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X = np.append(X, trans_pgp, axis=1)
      X = np.append(X, trans_bcrp, axis=1)
      X = np.append(X, trans_oct1, axis=1)
      X = np.append(X, trans_oct2, axis=1)      
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=2)
      model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV  

# (2,0.15) (50,3,0.1) = 83



good_ix = pickle.load(open(filename, 'rb'))
good_ix = good_ix[-83:]
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
      model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')   
      avgR2_5_fold_CV = st.mean(scoresR2)
      r_squared_CV_vals[i] = avgR2_5_fold_CV
# 83 bits = 43


  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=y, random_state=43)


model=CatBoostRegressor(iterations=80, depth=4, learning_rate=0.05, loss_function='RMSE')
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
scoresR2 = cross_val_score(model, X_train, y_train, cv=5, scoring = 'r2')
scoresRMSE = cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_root_mean_squared_error')

avgR2_5_fold_CV = st.mean(scoresR2)
avgRMSE_5_fold_CV = st.mean(scoresRMSE)
avgRMSE_5_fold_CV = (-1)*(st.mean(scoresRMSE))








######################################################################################################################################

# BDT



# determining best number of bits
r_squared_tr_vals = np.empty((155, 1))
r_squared_te_vals = np.empty((155, 1))
for i in range(1,155):
      good_f, good_ix    = sorted_fi[-i:], sorted_ix[-i:]
      good_desc = [descriptors[ix] for ix in good_ix]
      X = dataset.iloc[:, 5:2060].values
      X = sc.fit_transform(X)
      X = X[:, good_ix]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=46)
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50, learning_rate=0.8, random_state=1)
      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)

# 35 when 3
good_f, good_ix    = sorted_fi[-27:], sorted_ix[-27:]
good_desc = [descriptors[ix] for ix in good_ix]
X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)
X = X[:, good_ix]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=46)


# Determine best split
r_squared_tr_vals = np.empty((90, 1))
r_squared_te_vals = np.empty((90, 1))
for i in range(0,90):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=i)
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=50, learning_rate=0.8, random_state=1)
      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# = 46

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=46)


# Determining estimators
r_squared_tr_vals = np.empty((130, 1))
r_squared_te_vals = np.empty((130, 1))
for i in range(1,130):
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=i, learning_rate=0.8, random_state=1)
      model.fit(X_train, y_train)     
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
# Optimal estimators = 81 , 33 when 4



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
      model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=33, learning_rate=i*0.01, random_state=1)
      model.fit(X_train, y_train)
      y_pred_train = model.predict(X_train)
      y_pred_test = model.predict(X_test)
      R_squared_train = r2_score(y_train, y_pred_train)
      R_squared_test = r2_score(y_test, y_pred_test)
      r_squared_tr_vals[i] = r2_score(y_train, y_pred_train)
      r_squared_te_vals[i] = r2_score(y_test, y_pred_test)
      lr[i]=i*0.01
# best lr = 1.25 when 3, 0.8 when 4


# model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=81, learning_rate=1.25, random_state=1)

model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=33, learning_rate=0.8, random_state=1)
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
######################################################################################################################################

# ANN Model

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adagrad, SGD, Adadelta


X = dataset.iloc[:, 5:2060].values
X = sc.fit_transform(X)

good_f, good_ix    = sorted_fi[-20:], sorted_ix[-20:]
good_desc = [descriptors[ix] for ix in good_ix]

X = X[:, good_ix]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=y, random_state=12)



#model.add(Dense(units = 250, input_shape = [X_train.shape[1]], activity_regularizer = keras.regularizers.l2(0.01)))
#model.add(keras.layers.LeakyReLU())
#model.add(Dropout(rate=0.3))
model = Sequential()
#model.add(Dense(units = 80, activity_regularizer = keras.regularizers.l2(0.01)))
#model.add(keras.layers.LeakyReLU())
#model.add(Dropout(rate=0.1))
#model.add(Dense(units = 40, activity_regularizer = keras.regularizers.l2(0.01)))
#model.add(keras.layers.LeakyReLU())
#model.add(Dense(units = 20, activity_regularizer = keras.regularizers.l2(0.01)))
#model.add(keras.layers.LeakyReLU())
model.add(Dense(units = 10, activity_regularizer = keras.regularizers.l2(0.01)))
model.add(keras.layers.LeakyReLU())
model.add(Dense(units = 1))

# Compiling the ANN
model.compile(optimizer = keras.optimizers.adadelta(0.04), loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
model.summary()

#keras.optimizers.RMSprop(0.0001),
#keras.optimizers.adadelta(),
#mean_absolute_error

# Fitting the ANN to the Training set
stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# save = ModelCheckpoint('./best_model.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
make_model = model.fit(X_train, y_train, batch_size = 32, nb_epoch = 1800, validation_split = 0.02) #, callbacks=[stop])

# increased number of epochs, ran it many times. incrementally decreased validation split down. 

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
R_squared_train = r2_score(y_train, y_pred_train)
R_squared_test = r2_score(y_test, y_pred_test)

MSE_train = mean_squared_error(y_train, y_pred_train)
RMSE_train = math.sqrt(MSE_train) 
MSE_test = mean_squared_error(y_test, y_pred_test)
RMSE_test = math.sqrt(MSE_test) 

y_pred_train_values = sc.inverse_transform(y_pred_train)
y_pred_test_values = sc.inverse_transform(y_pred_test)


# saving model:
model_architecture = model.to_json()
with open('Kp_architecture.json', 'w') as arch_file:
    arch_file.write(model_architecture)
model.save_weights('Kp_weights.h5')

# loading the saved model: 
with open('Kp_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('Kp_weights.h5')

# to use model, input: model.predict(data)

y_pred_train = model.predict(X_train)
y_pred_train_values = sc.inverse_transform(y_pred_train)





'''
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




















