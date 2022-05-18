
import sqlite3
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

rawdata = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Desktop\pyt pro\Date_data.csv')
print(rawdata)
rawcopy = rawdata
cols = [0,3]
rawcopy = rawcopy.drop(rawcopy.columns[cols],axis=1)
rawcopy["count"] = rawcopy.groupby(['Consultant','Specialization']).transform('count')
rawcopy['com']=rawcopy['Complexity']
rawcopy.groupby(['Consultant','Specialization'])
no_of_cases = rawcopy.groupby(['Consultant', 'Specialization'], as_index=False).count()
no_of_cases.drop_duplicates(subset ="Consultant", keep = False, inplace = True)
table = pd.pivot_table(rawcopy,index='Consultant',columns='Complexity',values='com',aggfunc="count").reset_index()
table.drop_duplicates(subset ="Consultant", keep = False, inplace = True)
cols=[0,2,4]
no_of_cases.drop(no_of_cases.columns[cols],axis=1,inplace=True)
combined_data=table.join(no_of_cases)
copy_CD=combined_data
print(copy_CD)
d = {'1':'C1','2':'C2','3':'C3','4':'C4','5':'C5'}
copy_CD =copy_CD.rename(columns=lambda col: d.get(str(col)) if str(col) in d else col)
print(copy_CD)
copy_CD=copy_CD.fillna(0)
copy_CD['efficency'] = 0.2*copy_CD['C1']+0.4*copy_CD['C2']+0.6*copy_CD['C3']+0.8*copy_CD['C4']+1.0*copy_CD['C5']
copy_CD['Total Effeciency'] = copy_CD['efficency']/copy_CD['count'] 
copy_CD = copy_CD.rename(columns={'count':'Cases_Solved'})
print(copy_CD)
comp1 = copy_CD['C1'].values
comp2 = copy_CD['C2'].values
comp3 = copy_CD['C3'].values
comp4 = copy_CD['C4'].values
comp5 = copy_CD['C5'].values
case = copy_CD['Cases_Solved'].values
total_effeciency = copy_CD['Total Effeciency'].values
Cost_Len = len(comp1)
comp_eff = copy_CD[['C1','C2','C3','C4','C5','Cases_Solved']] 
X = comp_eff.values
Y = copy_CD.iloc[:, 8].values
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size = 0.75, test_size = 0.25, random_state = 0)
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
print(reg.intercept_)
print(reg.coef_)
y_pred = reg.predict(X_test) ; 
z_pred = reg.predict(X_train)
train_value = 10*(z_pred)
test_value = 10*(y_pred)
print(train_value)
print(test_value)
f = pd.DataFrame({'Actual': Y_train, 'Predicted': z_pred})
r = pd.DataFrame({'Actual': Y_test, 'Predicted':y_pred})
print(f)
print(r)
new_list = list()
new_list.append(train_value)
new_list.append(test_value)
Final_list = np.concatenate(new_list)
print (Final_list)
Final_table = pd.DataFrame({'id':copy_CD.Cases_Solved, 'Consultant':copy_CD.Consultant, 'Specialization':copy_CD.Specialization, 'Cases': copy_CD.Cases_Solved, 'Efficiency':Final_list.round(2)})
print(Final_table)
train_mean = z_pred.mean()
test_mean = y_pred.mean()
print('Mean of predicted train values:',train_mean)
print('Mean of predicted test values:',test_mean)
train_rms_value = np.sqrt(metrics.mean_squared_error(Y_train, z_pred))
test_rms_value = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))   
print('Root Mean Squared value for train data:',train_rms_value)
print('Root Mean Squared value for test data :', test_rms_value)
train_error_per = ((train_rms_value)/(train_mean))*100
test_error_per = ((test_rms_value)/(test_mean))*100
print('Percentage of Error for train data:' ,train_error_per,'%')
print('Percentage of Error for test data:',test_error_per,'%')