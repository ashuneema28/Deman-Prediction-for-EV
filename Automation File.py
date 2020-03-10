from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time
import glob
import os
import pandas as pd
import numpy as np
import joblib

chromedriver_location = "C:/webDrivers/chromedriver.exe"

driver = webdriver.Chrome(chromedriver_location)
driver.get("https://na.chargepoint.com/home")

username_input = '//*[@id="user_name"]'

password_input ='//*[@id="user_password"]'

log_in_submit = '//*[@id="validate-login"]'

reports = '//*[@id="mat-tab-label-0-3"]/div/a/span'

update_button = '//*[@id="updateChartBtn"]'

main_report = '//*[@id="view_select"]'

export_path ='//*[@id="export_select"]'

driver.find_element_by_xpath(username_input).send_keys("mshepherd5")
driver.find_element_by_xpath(password_input).send_keys("Parker#11")
driver.find_element_by_xpath(log_in_submit).click()
# driver.find_element_by_xpath(reports).click()
driver.implicitly_wait(5)

driver.find_element_by_xpath(reports).click()

driver.implicitly_wait(10)

iframe = driver.find_element_by_id('phpFrame')

driver.switch_to.frame(iframe)

select = Select(driver.find_element_by_xpath(main_report))

driver.implicitly_wait(10)

select.select_by_visible_text('Session Details Table')

time.sleep(8)

driver.find_element_by_xpath(update_button).click()

time.sleep(10.0)

select = Select(driver.find_element_by_xpath(export_path))

driver.implicitly_wait(15)

select.select_by_visible_text('Summary (csv)')

time.sleep(10)
list_of_files = glob.glob('C:/Users/A02290684/Downloads/*.csv') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print (latest_file)

append_df = pd.read_csv(latest_file)

df = pd.read_csv("C:/Users/A02290684/Desktop/Thesis/Main_DATA_FILE.csv")

df.append(append_df, ignore_index = True)
df=df[['Start Date','MAC Address','Charging Time (hh:mm:ss)','Energy (kWh)']]


df=df.sort_values(by='Start Date',ascending=True)

df["Start Date"]= pd.to_datetime(df["Start Date"])
df.set_index(['Start Date'])

df['Start Hour'] = df['Start Date'].dt.hour
df['Start Year'] = df['Start Date'].dt.year

'''Separating for only 2019 and 2020 values'''
options =["2019","2020"]
df=df.loc[df['Start Year'].isin(options)]


'''Separating for 6 stations with maximum historical data'''
options=['0024:B100:0002:4F68',
         '0024:B100:0002:596C',
         '0024:B100:0002:59ED',
         '0024:B100:0002:2BC3',
         '0024:B100:0002:EB54',
         '0024:B100:0002:52FF']

df=df.loc[df['MAC Address'].isin(options)]

df["Charging Time in minutes"] = df['Charging Time (hh:mm:ss)'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
df["Energy_per_hour"] = (df["Energy (kWh)"] / df["Charging Time in minutes"])*60


'''replacing inf and nan values with 0'''
df = df.replace([np.inf, -np.inf], np.nan)

df["Energy_per_hour"] = df["Energy_per_hour"].fillna(0)
'''replacement done'''

'''removing missing values'''
indexNames = df[ df['Energy_per_hour'] == 0].index

# Delete these row indexes from dataFrame
df.drop(indexNames, inplace=True)
'''Done removing values'''

df = df[['Start Date','Start Hour','Energy_per_hour']]

df["Start Date"]= pd.to_datetime(df["Start Date"])
df.set_index(['Start Date'])

df['Start Date'] = pd.to_datetime(df['Start Date']).dt.date

df = df.groupby(['Start Date','Start Hour'])[['Energy_per_hour']].sum()

df.to_csv("Data_Without_Context.csv")

'''Change the path to the relative path of file in your pc'''

df = pd.read_csv("C:/Users/A02290684/Desktop/Thesis/Prediction/First Test/Data_Without_Context.csv")

'''Change the path to the relative path of file in your pc'''
holiday = pd.read_csv('C:/Users/A02290684/Desktop/Thesis/Prediction/US Bank holidays.csv')
holiday['Holiday'] = pd.to_datetime(holiday['Holiday']).dt.date

df = df.merge(holiday, left_on = 'Start Date',right_on = 'Holiday',how = 'left')
df['holiday_ind'] = np.where(df['Holiday'].isna(),0,1)


df = df[['Start Date','Start Hour','Energy_per_hour','holiday_ind']]

# df.set_index(['Start Date'],inplace=True)
df["Start Date"]= pd.to_datetime(df["Start Date"])

df['Weekend'] = df['Start Date'].dt.dayofweek
df['Weekend'] = (df["Weekend"] > 4).astype(int)

df = df[['Start Date','Start Hour','Energy_per_hour','holiday_ind','Weekend']]

'''Assigning Previous value column'''
df['Prev_value'] = df.groupby("Start Hour")["Energy_per_hour"].shift(fill_value=0)

df.set_index(['Start Date'],inplace=True)
df.to_csv("Data_with_context.csv")
# print(df)

'''Prediction starts'''

'''Change the path to the relative path of file in your pc'''
original_df = pd.read_csv("C:/Users/A02290684/Desktop/Thesis/Prediction/First Test/Data_with_context.csv")

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

result_df.set_index(['Start Date'],inplace=True)