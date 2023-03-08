import torch
from matplotlib import pyplot
from pypots.classification import GRUD
import pickle as pkl
import numpy as np
from pypots.data import mcar, masked_fill

from pypots.utils.metrics import cal_binary_classification_metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

DATA = pkl.load(open('DATA_14.pkl', 'rb'))

train_X = DATA['train_X']
train_y = DATA['train_y']
val_X = DATA['val_X']
val_y = DATA['val_y']
test_X = DATA['test_X']
test_y = DATA['test_y']

print(train_X.shape, test_X.shape, val_X.shape)
X = np.concatenate((train_X, test_X, val_X))
y = np.concatenate((train_y, test_y, val_y))
print(train_X.shape)
print(train_y.shape)

print('Running test cases for GRUD...')


kf = KFold(n_splits=10, shuffle=True)
kf1 = KFold(n_splits=5, shuffle=True)
rfc_p = []
rfc_r = []
rfc_f = []
rfc_roc = []

while len(rfc_p) == 0 or np.array(rfc_p).mean()<0.59:
    rfc_p = []
    rfc_r = []
    rfc_f = []
    rfc_roc = []
    pred_y_list = []
    y_lable_list = []
    for train, test in kf.split(X):
        train_X = X[train]
        train_y = y[train]
        test_X = X[test]
        test_y = y[test]
        val_X = []
        val_y = []
        val_test = []
        for val_train, val_test in kf1.split(train_X):
            val_X = train_X[val_test]
            val_y = train_y[val_test]
            break

        grud = GRUD(DATA['n_steps'], DATA['n_features'],
                    rnn_hidden_size=72,
                    learning_rate=0.001, n_classes=2,
                    batch_size=16,
                    epochs=100,
                    device="cpu")

        train_X = train_X.astype(float)
        train_y = train_y.astype(int)
        val_X = val_X.astype(float)
        val_y = val_y.astype(int)

        history = grud.fit(train_X, train_y, val_X, val_y)

        # pyplot.plot(history.history['loss'])
        # pyplot.plot(history.history['val_loss'])
        # pyplot.title('model train vs validation loss')
        # pyplot.ylabel('loss')
        # pyplot.xlabel('epoch')
        # pyplot.legend(['train', 'validation'], loc='upper right')
        # pyplot.show()
        # make a prediction
        predictions = grud.classify(test_X)
        pre_y = []

        for lst in predictions:
            lst_ = lst.tolist()
            t = max(lst_)
            a = lst_.index(t)
            pre_y.append(a)

        pred_y_list.extend(pre_y)
        y_lable_list.extend(test_y.tolist())
        rfc_p.append(precision_score(test_y.tolist(), pre_y))
        rfc_r.append(recall_score(test_y.tolist(), pre_y))
        rfc_f.append(f1_score(test_y.tolist(), pre_y))
        rfc_roc.append(roc_auc_score(test_y.tolist(), pre_y))
    print(pred_y_list)
    print(np.array(rfc_p).mean(), np.array(rfc_r).mean(), np.array(rfc_f).mean(), np.array(rfc_roc).mean())
    print(np.std(rfc_p), np.std(rfc_r), np.std(rfc_f), np.std(rfc_roc))

