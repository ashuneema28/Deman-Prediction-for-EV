import pandas as pd
import numpy as np
import pickle
import joblib

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


# Xnew = scalarX.transform(Xnew)

ynew = model.predict(Xnew)
# show the inputs and predicted outputs
#for i in range(len(Xnew)):
#    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

result_df =  df_Copy[["Start Date","Start Hour"]]
result_df["Predicted Values"] =ynew
result_df.set_index(['Start Date'],inplace=True)

# result_df.to_csv("Prediction_Results.csv")
print(result_df)