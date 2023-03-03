import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y = pickle.load(open('y_phone_ring_14.pkl', 'rb'))

print(y)
y['BaseLine'] = ''
y['2weeks'] = ''
y['12weeks'] = ''
y['pre_12weeks'] = ''
y['label'] = ''
y['pre_label'] = ''

df = pd.read_excel('HAMD(1).xlsx')
for i, singal in y.iterrows():
    id = singal['Id']
    for index, row in df.iterrows():
        if row['研究对象'] == id:
            singal['BaseLine'] = row['Baseline']
            singal['2weeks'] = row['2weeks']
            singal['12weeks'] = row['12weeks']
            singal['label'] = 0 if row['Treatment Response at 12 weeks'] == 'Y' else 1
            break

for i, singal in y.iterrows():
    singal['pre_12weeks'] = max(0, 12*((singal['2weeks']-singal['BaseLine'])/2)+singal['BaseLine'])
    singal['pre_label'] = 0 if abs(singal['pre_12weeks']-singal['BaseLine'])/singal['BaseLine']>0.5 else 1

print(y)

rfc_p = precision_score(y['label'].tolist(), y['pre_label'].tolist())
rfc_r = recall_score(y['label'].tolist(), y['pre_label'].tolist())
rfc_f = f1_score(y['label'].tolist(), y['pre_label'].tolist())
rfc_roc = roc_auc_score(y['label'].tolist(), y['pre_label'].tolist())
print(rfc_p, rfc_r, rfc_f, rfc_roc)



y_28 = pickle.load(open('y_phone_ring_28.pkl', 'rb'))
y_28.rename(columns = {'Category':'C', 'Id':'Category'}, inplace = True)
y_28.rename(columns = {'C':'Id'}, inplace = True)


print(y_28)
y_28['BaseLine'] = ''
y_28['2weeks'] = ''
y_28['4weeks'] = ''
y_28['12weeks'] = ''
y_28['pre_12weeks'] = ''
y_28['label'] = ''
y_28['pre_label'] = ''

for i, singal in y_28.iterrows():
    id = singal['Id']
    for index, row in df.iterrows():
        if row['研究对象'] == id:
            singal['BaseLine'] = row['Baseline']
            singal['2weeks'] = row['2weeks']
            singal['4weeks'] = row['4weeks']
            singal['12weeks'] = row['12weeks']
            singal['label'] = 0 if row['Treatment Response at 12 weeks'] == 'Y' else 1
            break

for i, singal in y_28.iterrows():
    singal['pre_12weeks'] = max(0, 12*(
            ((2*singal['2weeks']+4*singal['4weeks'])-3*2*(singal['BaseLine']+singal['2weeks']+singal['4weeks'])/3)
            /((4+16)-6*4)
    )+singal['BaseLine'])
    singal['pre_label'] = 0 if abs(singal['pre_12weeks']-singal['BaseLine'])/singal['BaseLine']>0.5 else 1

print(y_28)
print(y_28['pre_label'].tolist())

rfc_p = precision_score(y_28['label'].tolist(), y_28['pre_label'].tolist())
rfc_r = recall_score(y_28['label'].tolist(), y_28['pre_label'].tolist())
rfc_f = f1_score(y_28['label'].tolist(), y_28['pre_label'].tolist())
rfc_roc = roc_auc_score(y_28['label'].tolist(), y_28['pre_label'].tolist())
print(rfc_p, rfc_r, rfc_f, rfc_roc)