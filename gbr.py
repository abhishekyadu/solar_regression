
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st


warnings.filterwarnings("ignore")
data = pd.read_csv("solarpowergeneration.csv")# reading csv file



data1 = data[70:]

#plt.figure(figsize=(12,4))
#sns.lineplot(x='Year', y= 'CO2', data = co2data1)

# loading the trained model
pickle_in = open('gbr.pkl', 'rb') 
model_gbr = pickle.load(pickle_in)

pickle_in = open('lasso.pkl', 'rb') 
model_lasso= pickle.load(pickle_in)

#train = co2data1[:101]
#test = co2data1[101:]
st.title("Given Solar Regression Data")
st.line_chart(data1)
st.sidebar.title("Solar Regression ML APP")
option = st.sidebar.selectbox('Select  period',
     ('5D', '10D', '15D', '20D'))
    
if option == '5D':
        f_period = 5 
if option == '10D':
        f_period = 10 
if option == '15D':
        f_period = 15 
if option == '20D':
        f_period = 20 


# Train the model
gbr.fit(x_train, y_train)
y_pred_train = gbr.predict(x_train)
y_pred_test = gbr.predict(x_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

print("R-squared on the training set:", r2_train)
print("R-squared on the testing set:", r2_test)
print("RMSE on the training set:", rmse_train)
print("RMSE on the testing set:", rmse_test)
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 120, 140],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
gbr = GradientBoostingRegressor(random_state=42)

# Set up the grid search
grid = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search
grid.fit(x_train, y_train)

def main():       
   
      
    # display the front end aspect
    #st.markdown(html_temp, unsafe_allow_html = True) 
    
    if st.sidebar.button("gbr Model"):
        future_df_gbr['forecast_gbr']= model_gbr.predict(start = future_df_gbr.index[145], end = 145 + f_period)   
        fpred = pd.DataFrame(future_df_gbr["forecast_gbr"][145:])
        st.title("Forecasted Solar Regresion  Data by gbr Model")
        st.dataframe(fpred)
        st.line_chart(future_df_gbr)
        
    if st.sidebar.button("lasso Model"):
        future_df_lasso['forecast_lasso']= model_lasso.predict(start = future_df_lasso.index[145], end = 145 + f_period)   
        fpred = pd.DataFrame(future_df_lasso["forecast_lasso"][145:])
        st.title("Forecasted CO2 Data by lasso Model")
        st.dataframe(fpred)
        st.line_chart(future_df_lasso)
        
    
    
    
if __name__=='__main__': 
    main()


