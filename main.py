import pandas as pd
from sklearn import model_selection as ms
from sklearn import tree as tr
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate as table
import sklearn.metrics as metrics
from sklearn.feature_selection import RFE
from sklearn.tree import export_text

#region 0. Retrieve data and configure option.

pd.set_option('display.max_columns', None)  # Configure pandas behaviour. Set display option to show all columns
# Read the file, upload the data and create the DataFrame with the columns selected for this job.
products = pd.read_excel(io="organics_updated.xlsx")[['ID','DemAffl', 'DemAge', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass', 'PromSpend', 'PromTime','TargetAmount']].copy()
#endregion

#region 1. DATA UNDERSTANDING

# # General examination
# print('Table preview: \n', products.head(), '\n')
# print('Total record count: ',products.shape[0])
# print('Total column count: ',products.shape[1])
#
# # Check for duplicated rows
# double_data = products[products.duplicated()]
# if double_data.shape[0] > 0:
#     print('Number of duplicated rows: ', double_data.shape[0])
# else:
#     print('Number of duplicated rows: ',double_data.shape[0])
#
# #Revise each column individually. We verify its Measures of Centrality.
# for column in products.columns:
#     if column != 'ID':
#         print('\n ---------------------------')
#         print('Column name:',column)
#         print('Data type:',products[column].dtype)
#         print('Missing values:', products[column].isnull().sum())
#         print('Unique values:', products[column].nunique())
#         if pd.api.types.is_numeric_dtype(products[column]):
#             print('Quantiles:')
#             print(products[column].quantile([0.15,0.25, 0.5, 0.75, 0.9]).to_string())
#             print('Mean:',products[column].mean())
#             print('Median:',products[column].median())
#             print('Minimum:',products[column].min())
#             print('Maximum:',products[column].max())
#             print('Standard Deviation:',products[column].std())
#             print('Mode:', products[column].mode())
#             print('---------------------------\n')
#             # if column == 'TargetAmount':
#             #     # Exploration of the values and distribution of the target variable
#             #     sns.displot(products["TargetAmount"], kde='True',bins=50,height=6)
#             #     sns.set_style('darkgrid')
#             #     plt.subplots_adjust(top=0.9)
#             #     plt.title("Distribution of Target Amount")
#             #     plt.show()
#         else:
#             print(products[column].describe())
#             for value, count in products[column].value_counts().items():
#                 print('Value -> Quantity :', value, '->', count)
#     else:
#         continue
# endregion

#region 2. DATA PREPARATION

#Visualization of the data to understand the nature of outliers and deciding whether they need to be deleted or not
# fig4, axes4 = plt.subplots(nrows=2,ncols=3)
# axes4 = axes4.flatten() # Flatten the axes array for easier iteration
# index4 = 0
# for column4 in ["DemAge","DemAffl","PromSpend","PromTime","TargetAmount"]:
#     sns.boxplot(products, y=column4 ,ax=axes4[index4])
#     index4 +=1
# fig4.subplots_adjust(bottom=0.3)
# plt.tight_layout()
# plt.show()

# products_cont = products.copy()[['DemAffl', 'DemAge', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass', 'PromSpend', 'PromTime','TargetAmount']]
# sns.pairplot(data=products_cont, hue='DemGender' )
# plt.show()
# del products_cont

# To perform necessary cleaning we check the number of missing values per column
print('Data Preparation Stage: \n Summary of Missing Values \n')
tmp = products.isnull().sum().reset_index()   # We get the result as a DataFrame instead of a Series
tmp.columns = ['Column Name', 'Missing Values']
tmp['Percentage']: float = round((tmp['Missing Values']/products.count().iloc[1])*100,2)
print(table.tabulate(tmp, headers=tmp.columns, tablefmt='simple_grid'))

# Update of DemClusterGroup . Impute the value 'U' to the missing values.
print('Missing values in DemClusterGroup : ',products['DemClusterGroup'].isna().sum())
products['DemClusterGroup'] = products['DemClusterGroup'].fillna('U')
print('Missing values in products after Imputation: ',products['DemClusterGroup'].isna().sum())

#Removal of the top three outliers in PromSpend
products_sorted = products.sort_values(by=['PromSpend'], ascending=False)
products = products_sorted.iloc[3:]

#Replace of the spaces in the text values like in the DemReg and DemTVReg
products.loc[:,'DemTVReg'] = products['DemTVReg'].str.replace(' & ', '', regex=False)
products.loc[:,'DemReg'] = products['DemReg'].str.replace(' & ', '', regex=False)
products.loc[:,'DemTVReg'] = products['DemTVReg'].str.replace('  ', ' ', regex=False)
products.loc[:,'DemReg'] = products['DemReg'].str.replace('  ', ' ', regex=False)
products.loc[:,'DemReg'] = products['DemReg'].str.replace(' ', '_', regex=False)
products.loc[:,'DemTVReg'] = products['DemTVReg'].str.replace(' ', '_', regex=False)

#region Plotting again to check the changes in the distribution of the columns after cleaning and imputation
# sns.displot(products["PromSpend"], kde='True',bins=50,height=6)
# sns.set_style('darkgrid')
# plt.subplots_adjust(top=0.9)
# plt.title("Distribution of Promotional Spend")
# plt.show()
#
# sns.boxplot(y='PromSpend', data=products)
# plt.show()
#
# products_cont = products.copy()[['DemAffl', 'DemAge', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass', 'PromSpend', 'PromTime','TargetAmount']]
# sns.pairplot(data=products_cont, hue='DemGender' )
# plt.show()
# del products_cont

#Plot to provide summary visualization for each predictor
# fig4, axes4 = plt.subplots(nrows=3,ncols=3, figsize=(15, 10))
# axes4 = axes4.flatten()
# for k, column4 in enumerate(['DemAffl', 'DemAge', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass', 'PromSpend', 'PromTime']):
#     if pd.api.types.is_numeric_dtype(products[column4]):
#         sns.histplot(products[column4],kde=True,ax=axes4[k-1])
#     else:
#         axes4[k-1].tick_params(axis='x', labelrotation=45)
#         sns.countplot(data=products, x=column4, ax=axes4[k - 1])
# fig4.subplots_adjust(bottom=0.3)
# plt.tight_layout()
# plt.show()

#endregion

#Separate features in dependent variable (Y) and predictors (X)
y = products["TargetAmount"]
x = products.drop(["TargetAmount","ID"],axis=1) # Drop by column name and get only
# Encoding of the nominal variables before partitioning
x = pd.get_dummies(x,columns=['DemGender', 'DemReg', 'DemTVReg', 'PromClass','DemClusterGroup'])
for column in x.columns:
    if x[column].dtype == 'bool':
        x[column] = x[column].astype(int)

# Verify the results obtained before proceeding to the partition and modeling
# print(x.shape)
# print(x.head())
# for name in x.columns:
#     print(name)

# Partitioning the data: 70% for training, 15% for testing and 15% for validation.
x_train, x_temp, y_train, y_temp = ms.train_test_split(x, y, test_size=0.3,train_size=0.7 , random_state=12445)
x_val, x_test, y_val, y_test = ms.train_test_split(x_temp, y_temp, test_size=0.5, random_state=12345)

print("Dimensions of Training Datasets",x_train.shape, y_train.shape)
print("Dimensions of Validation Datasets",x_val.shape, y_val.shape)
print("Dimensions of Test Datasets",x_test.shape, y_test.shape)

#endregion

#region 3. DATA MODELING

#region Run first DECISION TREE, Full Variable Model (Unpruned Decision Tree)
# regressor1 = tr.DecisionTreeRegressor(random_state=42)
# regressor1.fit(x_train, y_train)            # Train the model on the training data
# y_pred_test1 = regressor1.predict(x_test)   # Make predictions on the testing data
# y_pred_val1 = regressor1.predict(x_val)     # Make predictions on the validation data
#
# #Metrics
# mse_test1   = metrics.mean_squared_error(y_test, y_pred_test1) # Calculate the MSE
# r2_test1    = metrics.r2_score(y_test, y_pred_test1)  # Calculate R-squared
# mae_test1   = metrics.mean_absolute_error(y_test, y_pred_test1) # Calculate MAE
#
# mse_val1    = metrics.mean_squared_error(y_val, y_pred_val1) # Calculate the MSE
# r2_val1     = metrics.r2_score(y_val, y_pred_val1)  # Calculate R-squared
# mae_val1    = metrics.mean_absolute_error(y_val, y_pred_val1) # Calculate MAE

#endregion

#region Run second DECISION TREE, Manually Adjusted Decision Tree
parameters = {
    'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    'splitter' : ["best", "random"],
    'min_samples_split' : [2, 5, 10], # It is a percentage
    'max_depth':[1,2,3,4,5],
    'max_features' : ['sqrt','log2']
}
regressor_cv = tr.DecisionTreeRegressor(random_state=42)
cv = ms.GridSearchCV(regressor_cv,param_grid=parameters, cv=5)
cv.fit(x_train, y_train, ) # Train the model on the training data
print('Best parameters estimated: ', cv.best_params_)

cv.fit(x_train, y_train)
regressor_cv.fit(x_train, y_train)
y_pred_test2 = cv.predict(x_test)
y_pred_val2 = cv.predict(x_val)

#Metrics
mse_test2 = metrics.mean_squared_error(y_test, y_pred_test2) # Calculate the MSE
r2_test2 = metrics.r2_score(y_test, y_pred_test2)  # Calculate R-squared
mae_test2 = metrics.mean_absolute_error(y_test, y_pred_test2) # Calculate MAE

mse_val2 = metrics.mean_squared_error(y_val, y_pred_val2) # Calculate the MSE
r2_val2 = metrics.r2_score(y_val, y_pred_val2)  # Calculate R-squared
mae_val2 = metrics.mean_absolute_error(y_val, y_pred_val2) # Calculate MAE
#endregion

#region Run third DECISION TREE. This one has the features pre-selected using recursive feature elimination (RFE).

# regressor3 = tr.DecisionTreeRegressor(random_state=42)
# rfe = RFE(estimator=regressor3, n_features_to_select=14)
# x_train_selected = rfe.fit_transform(x_train, y_train)
# x_test_selected = rfe.transform(x_test)
# x_val_selected = rfe.transform(x_val)
#
# regressor3.fit(x_train_selected, y_train) # Train the model on the selected features
#
# # Make predictions on the testing and validation data that have been transformed (selected)
# y_pred_test3 = regressor3.predict(x_test_selected)
# y_pred_val3  = regressor3.predict(x_val_selected)
#
# #Metrics of the Decision Tree
#
# mse_test3     = metrics.mean_squared_error(y_test, y_pred_test3)    # Calculate the MSE
# r2_test3      = metrics.r2_score(y_test, y_pred_test3)    # Calculate R-squared
# mae_test3     = metrics.mean_absolute_error(y_test, y_pred_test3)   # Calculate MAE
#
#
# mse_val3      = metrics.mean_squared_error(y_val, y_pred_val3) # Calculate the MSE
# r2_val3       = metrics.r2_score(y_val, y_pred_val3)  # Calculate R-squared,
# mae_val3      = metrics.mean_absolute_error(y_val, y_pred_val3) # Calculate MAE,
#
# # Get the names of the selected features and print them
# print(f"The {20} Selected Features Were:", x.columns[rfe.get_support(indices=True)],"\n")

#endregion

#region Run fourth DECISION TREE. Use the Pre-selected features from the previous step and the parameters for pre-pruning it.

# regressor4 = tr.DecisionTreeRegressor(**cv.best_params_, random_state=42) # and prune it with the estimated parameters
# regressor4.fit(x_train_selected, y_train) # train the model with the features selected
#
# # Make predictions on the testing and validation data that have been transformed (selected)
# y_pred_test4 = regressor4.predict(x_test_selected)
# y_pred_val4 = regressor4.predict(x_val_selected)
#
# #Metrics
# mse_test4 = metrics.mean_squared_error(y_test, y_pred_test4)    # Calculate the MSE
# r2_test4 = metrics.r2_score(y_test, y_pred_test4)               # Calculate R-squared
# mae_test4 = metrics.mean_absolute_error(y_test, y_pred_test4)   # Calculate MAE
#
#
# mse_val4 = metrics.mean_squared_error(y_val, y_pred_val4) # Calculate the MSE
# r2_val4 = metrics.r2_score(y_val, y_pred_val4)  # Calculate R-squared
# mae_val4 = metrics.mean_absolute_error(y_val, y_pred_val4) # Calculate MAE


#endregion

#region Table of Results
# data = [
# {"No.":1,"Tree Type":"Unpruned","Adjustment":"None","Dataset": "Test",       "MSE": round(mse_test1,3) ,"R^2": round(r2_test1,3),"MAE": round(mae_test1,3)},
# {"No.":1,"Tree Type":"Unpruned","Adjustment":"None","Dataset": "Validation", "MSE": round(mse_val1,3) , "R^2": round(r2_val1,3), "MAE": round(mae_val1,3)},
#
# {"No.":2,"Tree Type":"Pruned","Adjustment":"Estimated Pruning Parameters","Dataset": "Test",       "MSE": round(mse_test2,3) ,"R^2": round(r2_test2,3),"MAE": round(mae_test2,3)},
# {"No.":2,"Tree Type":"Pruned","Adjustment":"Estimated Pruning Parameters","Dataset": "Validation", "MSE": round(mse_val2,3) , "R^2": round(r2_val2,3), "MAE": round(mae_val2,3)},
#
# {"No.":3,"Tree Type":"Unpruned","Adjustment":"RFE (Top "+str(14)+")","Dataset":"Test","MSE": round(mse_test3, 3), "R^2":round(r2_test3, 3), "MAE":round(mae_test3, 3)},
# {"No.":3,"Tree Type":"Unpruned","Adjustment":"RFE (Top "+str(14)+")","Dataset":"Test","MSE": round(mse_val3, 3), "R^2":round(r2_val3, 3), "MAE":round(mae_val3, 3)},
#
# {"No.":4,"Tree Type":"Pruned","Adjustment":"RFE + Pre-pruned","Dataset":"Test",      "MSE":round(mse_test4, 3), "R^2":round(r2_test4, 3), "MAE":round(mae_test4, 3)},
# {"No.":4,"Tree Type":"Pruned","Adjustment":"RFE + Pre-pruned","Dataset":"Validation","MSE":round(mse_val4, 3), "R^2":round(r2_val4, 3), "MAE":round(mae_val4, 3)}
# ]
#
# print(table.tabulate(data, headers="keys", tablefmt="simple_grid"))
#endregion

#endregion

########################################################################################
#region SCORING: Make some predictions with one of the Trained Models of my choice.

# Accessing the document where the values for predicting are stored
predict_amount = pd.read_excel(io="Documents/organics_score.xlsx")[['DemAffl', 'DemAge', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass', 'PromSpend', 'PromTime','TargetAmt']].copy()

#Replace of the spaces in the text values like in the DemReg and DemTVReg
predict_amount.loc[:,'DemTVReg'] = predict_amount['DemTVReg'].str.replace(' & ', '', regex=False)
predict_amount.loc[:,'DemReg']   = predict_amount['DemReg'].str.replace(' & ', '', regex=False)
predict_amount.loc[:,'DemTVReg'] = predict_amount['DemTVReg'].str.replace('  ', ' ', regex=False)
predict_amount.loc[:,'DemReg']   = predict_amount['DemReg'].str.replace('  ', ' ', regex=False)
predict_amount.loc[:,'DemReg']   = predict_amount['DemReg'].str.replace(' ', '_', regex=False)
predict_amount.loc[:,'DemTVReg'] = predict_amount['DemTVReg'].str.replace(' ', '_', regex=False)


# Separating the dependent variable from the predictors
y_predict = predict_amount["TargetAmt"]
x_predict = predict_amount.drop(["TargetAmt"],axis=1)

# Encoding of the nominal variables to Dummy
x_predict = pd.get_dummies(x_predict,columns=['DemGender', 'DemReg', 'DemTVReg', 'PromClass','DemClusterGroup'])
for column in x_predict.columns:
    if x_predict[column].dtype == 'bool':
        x_predict[column] = x_predict[column].astype(int)

#### Adding the missing columns that are used in the prediction but are not present in the new dataset
train_feature_names = x.columns

# Add missing dummy variables (if any) with a value of 0
for feature in train_feature_names:
    if feature not in x_predict.columns:
        x_predict[feature] = 0

# Remove extra dummy variables (if any)
x_predict = x_predict[train_feature_names]

# Running the prediction with the Decision Tree that uses the best Estimated Parameters
y_predict = cv.best_estimator_.predict(x_predict)


# Convert y_predict (NumPy array) into a Pandas Series with an appropriate column name
y_predict_series = pd.Series(y_predict, index=x_predict.index, name="TargetAmt")
predicted_df = pd.concat(objs=[x_predict, y_predict_series], axis=1)
print(predicted_df)

plt.figure(figsize=(24, 16))  # Adjust figure size as needed
tr.plot_tree(cv.best_estimator_,
          feature_names=train_feature_names,  # Replace x with your feature data
          class_names=['0', '1'],  # Adjust class names if needed
          filled=True,
          rounded=True,
          fontsize=11)  # Adjust fontsize as needed
plt.title("Prediction of Target Amount with a Decision Tree Regressor")
plt.show()

tree_rules = export_text(cv.best_estimator_, feature_names=train_feature_names.tolist())
print(tree_rules)


#endregion