# python
# split into input and outputs
import pickle as pkl

import torch
from keras import Sequential
from keras.layers import LSTM, Dense, GRU
import numpy as np
from numpy import concatenate
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DATA = pkl.load(open('DATA.pkl', 'rb'))
train_X = DATA['train_X']
train_y = DATA['train_y']
val_X = DATA['val_X']
val_y = DATA['val_y']
test_X = DATA['test_X']
test_y = DATA['test_y']

train_y[train_y == 1] = 0
val_y[val_y == 1] = 0
test_y[test_y == 1] = 0

train_y[train_y == 2] = 1
val_y[val_y == 2] = 1
test_y[test_y == 2] = 1

train_X = np.nan_to_num(train_X)
val_X = np.nan_to_num(val_X)
test_X = np.nan_to_num(test_X)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 28, -1))
test_X = test_X.reshape((test_X.shape[0], 28, -1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
X = np.concatenate((train_X, test_X, val_X))
y = np.concatenate((train_y, test_y, val_y))

# design network
model = Sequential()
model.add(GRU(72, input_shape=(train_X.shape[1], train_X.shape[2]), bias_constraint=True))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='sgd')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_data=(val_X, val_y), verbose=2,
                    shuffle=False)

kf = KFold(n_splits=10)

rfc_p = []
rfc_r = []
rfc_f = []
rfc_roc = []

for train, test in kf.split(X):
    train_X = X[train]
    train_y = y[train]
    test_X = X[test]
    test_y = y[test]
    val_X = []
    val_y = []
    for val_train, val_test in kf.split(X):
        val_X = train_X[val_test]
        val_y = train_y[val_test]
        break

    history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)

    # make a prediction
    yhat = model.predict(test_X)
    print(test_y)
    print(np.around(yhat.reshape(-1)).astype(int))

    pre_y = []
    for i in np.around(yhat.reshape(-1)).astype(int):
        if i > 1:
            pre_y.append(1)
        elif i < 0:
            pre_y.append(0)
        else:
            pre_y.append(i)

    rfc_p.append(precision_score(test_y, pre_y))
    rfc_r.append(recall_score(test_y, pre_y))
    rfc_f.append(f1_score(test_y, pre_y))
    rfc_roc.append(roc_auc_score(test_y, pre_y))
print(np.array(rfc_p).mean(), np.array(rfc_r).mean(), np.array(rfc_f).mean(), np.array(rfc_roc).mean())
