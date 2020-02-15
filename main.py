# Driver code
import pandas as pd
import h5py, datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_profiling as pp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE
import warnings
import build_model, feature_selector, preprocessing, train_model, create_sub
warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv('ChartbustersParticipantsData/Data_Train.csv')
test_df = pd.read_csv('ChartbustersParticipantsData/Data_Test.csv')

UniqueValues(train_df)
Views = train_df[['Views']]

df = train_df.drop(['Views'], 1)
df = df.append(test_df, ignore_index=True)

df['age'] = df['Timestamp'].map(lambda x: preprocessing.convert_time(x))
df['Likes2'] = df['Likes'].map(lambda x: int(preprocessing.process_likes(x)))
df['Popularity2'] = df['Popularity'].map(lambda x: int(preprocessing.process_likes(x)))

ndf = preprocessing.LabelEncoding(df)

ntrain_df = ndf[:78458]
ntest_df = ndf[78458:]

ntrain_df['Views'] = Views


X = ntrain_df.drop(['Views'], axis = 1)
y = ntrain_df[["Views"]]

features = feature_selector.feature_selection(X,y)


X_train = ntrain_df[features]
y_train = y
x_val = ntest_df[features]

train_size = int(len(y_train) * 0.80)
with h5py.File("dataset-v4.h5", 'w') as f:
        f.create_dataset("X_train", data=np.array(X_train[:train_size]))
        f.create_dataset('y_train', data=np.array(y_train[:train_size]))
        f.create_dataset("X_val", data=np.array(X_train[train_size:]))
        f.create_dataset("y_val", data=np.array(y_train[train_size:]))


history = train_model.train()

create_sub.sub(x_val, test_df)
