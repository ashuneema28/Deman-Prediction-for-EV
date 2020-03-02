import pandas as pd
import numpy as np
import pickle
import joblib
import random

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

# dictionary of lists

original_df = pd.read_csv("Grouped_data_With_Prev_Value_Column.csv")

dict = {'Start Date':["12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020","12/1/2020"],
        'Start Hour': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
        }

df = pd.DataFrame(dict)
df_Copy =pd.DataFrame(dict)

df['Start Date'] = pd.to_datetime(df['Start Date']).dt.date

holiday = pd.read_csv('C:/Users/A02290684/Desktop/Thesis/Prediction/US Bank holidays.csv')
holiday['Holiday'] = pd.to_datetime(holiday['Holiday']).dt.date

df = df.merge(holiday, left_on = 'Start Date',right_on = 'Holiday',how = 'left')
df['holiday_ind'] = np.where(df['Holiday'].isna(),0,1)


df = df[['Start Date','Start Hour','holiday_ind']]

# df.set_index(['Start Date'],inplace=True)
df["Start Date"]= pd.to_datetime(df["Start Date"])

df['Weekend'] = df['Start Date'].dt.dayofweek
df['Weekend'] = (df["Weekend"] > 4).astype(int)

df = df[['Start Date','Start Hour','holiday_ind','Weekend']]


vals = df['Start Hour'].values
lists =[]
for val in vals:
    # print(val)
    df2 = original_df[original_df['Start Hour'] == val]
    rows = np.random.choice(df2.index.values, 1)
    sampled_df = df2.loc[rows]
    a =sampled_df['Prev_value'].values[0]
    lists.append(a)

df['Prev_value']=lists
# df['Prev_value']=df['Prev_value'].replace(0,5)
# df['Prev_value']=df['Prev_value'].replace(0,9)

df.set_index(['Start Date'],inplace=True)

# model=pickle.load(open("EV_Predictor.sav", 'rb'))
model = joblib.load("EV_Predictor.sav")

# df.to_csv("Data_With_Holiday__And_Weekend_Index.csv")
dataset_2 = df.values
Xnew = dataset_2[:,0:]



ynew = model.predict(Xnew)

result_df =  df_Copy[["Start Date","Start Hour"]]
result_df["Predicted Values"] =ynew

'''Adding Visitors'''
number_of_people = [1, 1, 1, 1, 1, 3, 5, 8, 11, 9, 8, 9, 10, 10, 9, 10, 8, 9, 8, 6, 4, 2, 2, 1]
number_of_people = np.asarray(number_of_people)
result_df["Number of visitors"] = number_of_people

visitors = []
for x in range(0,len(number_of_people)):
    if(number_of_people[x]==1):
        test = np.random.choice(np.arange(0, 2), p=[0.8, 0.2])
        visitors.append(test)
    elif(number_of_people[x]==2):
        test = np.random.choice(np.arange(0, 3), p=[0.7, 0.1, 0.2])
        visitors.append(test)
    elif(number_of_people[x]==3):
        test = np.random.choice(np.arange(0, 4), p=[0.5, 0.1, 0.1, 0.3])
        visitors.append(test)
    elif(number_of_people[x]==4):
        test = np.random.choice(np.arange(0, 5), p=[0.2, 0.1, 0.2, 0.2, 0.3])
        visitors.append(test)
    elif(number_of_people[x]==5):
        test = np.random.choice(np.arange(0, 6), p=[0.05, 0.05, 0.05, 0.25, 0.25, 0.35])
        visitors.append(test)
    elif(number_of_people[x]==6):
        test = np.random.choice(np.arange(0, 7), p=[0.034, 0.033, 0.033, 0.1, 0.2, 0.3, 0.3])
        visitors.append(test)
    elif(number_of_people[x]==7):
        test = np.random.choice(np.arange(0, 8), p=[0.034, 0.033, 0.033, 0.1, 0.2, 0.2, 0.2, 0.2])
        visitors.append(test)
    elif(number_of_people[x]==8):
        test = np.random.choice(np.arange(0, 9), p=[0.02, 0.02, 0.02, 0.04, 0.1, 0.2, 0.2, 0.2, 0.2])
        visitors.append(test)
    elif(number_of_people[x]==9):
        test = np.random.choice(np.arange(0, 10), p=[0.02, 0.02, 0.02, 0.02, 0.05, 0.07, 0.2, 0.2, 0.2, 0.2])
        visitors.append(test)
    elif(number_of_people[x]==10):
        test = np.random.choice(np.arange(0, 11), p=[0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2])
        visitors.append(test)
    elif(number_of_people[x]==11):
        test = np.random.choice(np.arange(0, 12), p=[0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.2])
        visitors.append(test)
result_df["Number of probable visitors"] = visitors


result_df.loc[result_df['Number of probable visitors'] == 0, 'Predicted Values'] = result_df['Predicted Values']*0
result_df.loc[result_df['Number of probable visitors'] == 1, 'Predicted Values'] = result_df['Predicted Values']*1.0
result_df.loc[(result_df['Number of probable visitors'] > 1 ) & (result_df['Number of probable visitors'] <= 4), 'Predicted Values'] = result_df['Predicted Values']*1.5
result_df.loc[(result_df['Number of probable visitors'] >= 5 ) & (result_df['Number of probable visitors'] <= 8), 'Predicted Values'] = result_df['Predicted Values']*2.0
result_df.loc[(result_df['Number of probable visitors'] >= 9 ) & (result_df['Number of probable visitors'] <= 11), 'Predicted Values'] = result_df['Predicted Values']*2.5
# result_df.loc[result_df['Number of probable visitors'] >10 , 'Predicted Values'] = result_df['Predicted Values']*2.3



result_df = result_df[['Start Date','Start Hour','Number of probable visitors','Predicted Values']]
print(result_df)

result_df = result_df[['Start Date','Start Hour','Number of probable visitors','Predicted Values']]
result_df.set_index(['Start Date'],inplace=True)
# result_df.to_csv("Prediction_Results.csv")
