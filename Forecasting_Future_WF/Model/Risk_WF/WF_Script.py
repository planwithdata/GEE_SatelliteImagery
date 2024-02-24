#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[5]:

pd.set_option("display.max_rows", None)

# In[6]:

# Get Current Directory
import os
os.getcwd()


# In[7]:


os.chdir('C:\\Risk_WF\\Model_Files\\Training_Data')

current_taz_data_3 = gpd.read_file(r'ModelTaz.shp')

current_taz_data_2 = gpd.read_file(r'TAZ_By_LY_WF.shp')

current_taz_data = pd.read_excel(r'TAZ_By_LY_WF_excel_2.xlsx')


## Naming Columns to match
current_taz_data.rename (columns = {'TAZ':'Model_TAZ'},inplace = True)


current_taz_data_3.rename (columns = {'TOPOP_17':'TOT_POP','HH_17':'HH'},inplace = True)

current_taz_data_3.iloc[:,[1,9,17]]

current_taz_data_new=pd.merge(current_taz_data,current_taz_data_3.iloc[:,[1,9,17]],on = 'Model_TAZ', how = 'inner')


# In[22]:
## Feature Creation : Population Density
current_taz_data_new['POP_Den'] = current_taz_data_new['TOT_POP']/current_taz_data_new['ACRES']



# In[262]:


x = current_taz_data_new.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,25,26]]  


x = x.replace(np.nan, 0)

y = current_taz_data_new.iloc[:,[-7]] ## Summarized Areas



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size = 0.3,random_state = 2)


# Train the model with the best hyperparameters on the entire training data
regressor = RandomForestRegressor(n_estimators=800, random_state=42)
regressor.fit(x_train, y_train)


# In[321]:


# ## For Training Data

# r_square=regressor.score(x_train, y_train)
# r2_n=len(y_train)
# r2_k=x_train.shape[1]
# adjusted_r_square = 1 - (1-r_square)*(r2_n-1)/(r2_n-r2_k-1)
# print(r_square)
# print(adjusted_r_square)


# # In[322]:


# output = x_train
# y_pred_train = regressor.predict(x_train)
# np.set_printoptions(precision=2)
# output['y_pred_train']=y_pred_train
# output['y_train']=y_train
# output.iloc[:,[-1,-2]].head(10)


# # In[323]:


# ## For Training Data

# from sklearn.metrics import mean_squared_error

# mse = mean_squared_error(output.iloc[:,[-1]], output.iloc[:,[-2]])  ## (y_true, y_pred)
# rmse = np.sqrt(mse)
# print("RMSE (Training Data) = ",rmse)


# # In[310]:


# std = output.iloc[:,[-1]].std()
# print("StdDev_TrainData = ",std)


# # In[311]:


# y_max = output.iloc[:,[-1]].max()
# print("y_max = ",y_max)
# y_min = output.iloc[:,[-1]].min()
# print("y_min = ",y_min)
# nrmse = rmse/(y_max - y_min)  # Normalized RMSE for comparison with other models
# print("Normalized RMSE (Training Data)= ", nrmse)


# In[324]:


x_train = x_train.iloc[:,0:15]


# # In[325]:


# x_train.columns


# # In[211]:


# ### Need to make a better split that is a better representation of the whole data


# # In[326]:


# # For Test Data

# r_square=regressor.score(x_test, y_test)
# r2_n=len(y_test)
# r2_k=x_test.shape[1]
# adjusted_r_square = 1 - (1-r_square)*(r2_n-1)/(r2_n-r2_k-1)
# print(r_square)
# print(adjusted_r_square)


# # In[317]:


# x_test


# # In[331]:


# #x_test = x_test.iloc[:,0:15]


# # In[343]:


# x_test = x_test.iloc[:,0:15]
# output=x_test
# y_pred_test = regressor.predict(x_test)
# np.set_printoptions(precision=2)
# output['y_pred_test']=y_pred_test
# output['y_test']=y_test
# output.iloc[:,[-1,-2]].head(30)


# # In[274]:


# # For Test Data

# from sklearn.metrics import mean_squared_error

# mse = mean_squared_error(output.iloc[:,[-1]], output.iloc[:,[-2]])  ## (y_true, y_pred)
# rmse = np.sqrt(mse)
# print("RMSE (Test Data)= ",rmse)


# # In[275]:


# std = output.iloc[:,[-1]].std()
# print("StdDev_TestData = ",std)


# # In[276]:


# y_max = output.iloc[:,[-1]].max()
# print("y_max = ",y_max)


# # In[277]:


# y_min = output.iloc[:,[-1]].min()
# print("y_min = ",y_min)


# # In[278]:


# nrmse = rmse/(y_max - y_min)  # Normalized RMSE for comparison with other models
# print("Normalized RMSE (Test Data) = ", nrmse)  # Normalized RMSE for comparison with other models


# In[279]:


# # Obtain feature importances
# feature_importances = regressor.feature_importances_
# # Printing Variable Coefficients
# for feature, importance in zip(x_train.columns, feature_importances):
#     print(f"{feature}: {importance}")



x_test = x_test.iloc[:,0:15]



# os.chdir('C:\\Users\\rsingh\\OneDrive - PlanRVA\\Documents\\SP_ResiliencyModel\\Taz_2050 File\\Model50Taz')

# In[130]:

os.chdir('C:\\Risk_WF\\Model_Files\\Paste_Input_File_Here\\Base TAZ SHP')
Base_TAZ = gpd.read_file(r'Base_TAZ.shp')
Base_TAZ.columns

os.chdir('C:\\Risk_WF\\Model_Files\\Paste_Input_File_Here')
input_data = pd.read_excel(r'Input_WF.xlsx')
#input_data = pd.read_excel(r'RSLAM_Mock_Resiliency_WF_Input.xlsx')
#input_data.head()

if pd.notna(input_data.columns[3]):
    try:
        numeric_value = pd.to_numeric(input_data.columns[3])
        if not np.isnan(numeric_value):
            input_data.columns = input_data.iloc[0, :]
            input_data = input_data.drop(index=0)
        else:
            pass  # No action is needed
        
    except ValueError:
        pass  # No action is needed
else:
    input_data.columns = input_data.iloc[0, :]
    input_data = input_data.drop(index=0)


#input_data.head()

input_data = input_data.replace(np.nan, 0)

input_data['TAZ'] = input_data['TAZ'].astype(int)

Base_TAZ = Base_TAZ.replace(np.nan, 0)

Base_TAZ['TAZ'] = Base_TAZ['TAZ'].astype(int)

#print(input_data.dtypes)

forecasted_taz_data = pd.merge(Base_TAZ, input_data, on = 'TAZ', how = 'inner')

# forecasted_taz_data.columns

#forecasted_taz_data.head()

#forecasted_taz_data.columns

forecasted_taz_data['POP_Den'] = forecasted_taz_data['TOT_POP']/forecasted_taz_data['TAZ_ACRES_Ar']
#x_test = forecasted_taz_data.iloc[:,[8,9,10,11,12,13,14,15,16,17,18,19,20,-8,-1]]

x_test = pd.DataFrame()

x_test['L_D_RE'] = forecasted_taz_data['L_D_RE_Ar']  
x_test['M_D_RE_SF'] = forecasted_taz_data['M_D_RE_SF_Ar']  
x_test['M_D_RE_MF'] = forecasted_taz_data['M_D_RE_MF_Ar']  
x_test['H_D_RE'] = forecasted_taz_data['H_D_RE_Ar']  
x_test['MU'] = forecasted_taz_data['MU_Ar']  
x_test['COM'] = forecasted_taz_data['COM_Ar']  
x_test['INS'] = forecasted_taz_data['INS_Ar']  
x_test['OF'] = forecasted_taz_data['OF_Ar']  
x_test['IND'] = forecasted_taz_data['IND_Ar']  
x_test['Other'] = forecasted_taz_data['Other_Ar']  
x_test['AG'] = forecasted_taz_data['AG_Ar']  
x_test['FO'] = forecasted_taz_data['FO_Ar']  
x_test['PA'] = forecasted_taz_data['PA_Ar']  
x_test['HH'] = forecasted_taz_data['HH']  
x_test['POP_Den'] = forecasted_taz_data['POP_Den']  


#x_test.columns

y_pred = regressor.predict(x_test)

forecasted_taz_data['Future_Summ_Area'] = y_pred

forecasted_taz_data['Future_Acr_Aff'] = (forecasted_taz_data['Future_Summ_Area']*100)/forecasted_taz_data['TAZ_ACRES_Ar'] 
## 'Future_Acr_Aff' : Percentage of total TAZ ACRES that are Wildfire Prone
## 'Future_Acr_Aff_Prop' : Ratio of total TAZ ACRES that are Wildfire Prone - used to calculate other variables. (Scn_TOT_POP: ['Future_ACR_Aff_Prop'] x TOT_POP)
forecasted_taz_data['Future_ACR_Aff_Prop']=forecasted_taz_data['Future_Summ_Area']/forecasted_taz_data['TAZ_ACRES_Ar']

#forecasted_taz_data.iloc[:,[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]

# In[69]:


for i in range(35):

    #col = forecasted_taz_data.iloc[:,8+i]
    if forecasted_taz_data.columns[i+8] == 'JUR':
        continue
    elif ((forecasted_taz_data.columns[i+8] == 'TOT_POP')|("EMP" in forecasted_taz_data.columns[i+8])|("HH" in forecasted_taz_data.columns[i+8])):
        forecasted_taz_data['Scn_'+forecasted_taz_data.columns[i+8]] = round(forecasted_taz_data.iloc[:,8+i].astype(float)*forecasted_taz_data['Future_ACR_Aff_Prop'],0)
    else:
        pass
        
    #forecasted_taz_data['Scn_'+forecasted_taz_data.columns[i+8]] = forecasted_taz_data.iloc[:,8+i].astype(float)*forecasted_taz_data['Future_ACR_Aff_Prop']




# In[73]:


#Scn_Output = forecasted_taz_data.drop( columns = ['Scn_Parcel_ACR_Ar','Scn_TAZ_ACRES_Ar'])
Scn_Output = forecasted_taz_data



# In[85]:
os.chdir('C:\\Risk_WF\\Model_Files\\Output_Files\\Output_Shp')


# In[86]:


Scn_Output.to_file(driver = 'ESRI Shapefile', filename =r'Scn_Output.shp', crs = "EPSG:4326" )


# In[87]:


Scn_Output.to_excel('Scn_Output.xlsx')

#fig, ax = plt.subplots(figsize=(12, 10))
#forecasted_taz_data.plot(column='Future_Acr_Aff', legend=True, ax=ax)

# fig, ax = plt.subplots(figsize=(12, 10))
# forecasted_taz_data.plot(column='Future_Summ_Area', legend=True, ax=ax)