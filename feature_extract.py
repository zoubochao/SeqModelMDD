# import pickle as pkl
# import matplotlib.pyplot as plt
# from tsfresh.utilities.dataframe_functions import impute
#
# X_date = pkl.load(open('X_date_14.pkl', 'rb'))
# y = pkl.load(open('y_14.pkl', 'rb'))
#
# y['Category'] = y['Category'].apply(lambda x:  0 if x == 1 else x)
# y['Category'] = y['Category'].apply(lambda x:  1 if x == 2 else x)
# y.set_index(['Id'], inplace = True)
#
# X_date['Category'] = X_date['Id'].apply(lambda x: y.loc[x,:]['Category'])
# X_date = X_date.drop('date', axis = 1)
#
# X_date = X_date.fillna(value=0)
# # list = []
# #
# # for i in range(175):
# #     list.append(X_date.iloc[i*14 : i*14+14, ])
# #
# # print(list[0])
# #
# # for sample in list[:40]:
# #     plt.plot(sample['Unnamed: 0'], sample['T_1'],
# #                 color="mediumpurple" if sample['Id'].unique() == 0 else 'burlywood',
# #                 linestyle="-",
# #                 linewidth=1.0)
# #
# # plt.show()
# #
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures,load_robot_execution_failures
from tsfresh import extract_features,select_features
import pandas as pd
from tsfresh.utilities.dataframe_functions import impute

if __name__ == '__main__':
    timeseries, y = load_robot_execution_failures(file_name = 'lp1.data.txt')
    extracted_features = extract_features(timeseries,column_id = "id",column_sort = "time")
    print(timeseries)
    print(y)

    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    print(features_filtered)
