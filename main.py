import pandas as pd
from sklearn import model_selection as ms
from sklearn import tree as tr
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate as table
from sklearn.compose import ColumnTransformer
import sklearn.metrics as metrics


#region 0. Retrieve data and configure option.

pd.set_option('display.max_columns', None)  # Configure pandas behaviour. Set display option to show all columns
# Read the file, upload the data and create the DataFrame with the columns selected for this job.
products = pd.read_excel(io="organics_updated.xlsx")[['ID','DemAffl', 'DemAge', 'DemClusterGroup', 'DemGender', 'DemReg', 'DemTVReg', 'PromClass', 'PromSpend', 'PromTime','TargetAmount']].copy()
#endregion

#region 1. DATA UNDERSTANDING

# General examination
print('Table preview: \n', products.head(), '\n')
print('Total record count: ',products.shape[0])
print('Total column count: ',products.shape[1])

# Check for duplicated rows
double_data = products[products.duplicated()]
if double_data.shape[0] > 0:
    print('Number of duplicated rows: ', double_data.shape[0])
else:
    print('Number of duplicated rows: ',double_data.shape[0])

#Revise each column individually. We verify its Measures of Centrality.
for column in products.columns:
    if column != 'ID':
        print('\n ---------------------------')
        print('Column name:',column)
        print('Data type:',products[column].dtype)
        print('Missing values:', products[column].isnull().sum())
        print('Unique values:', products[column].nunique())
        if pd.api.types.is_numeric_dtype(products[column]):
            print('Quantiles:')
            print(products[column].quantile([0.15,0.25, 0.5, 0.75, 0.9]).to_string())
            print('Mean:',products[column].mean())
            print('Median:',products[column].median())
            print('Minimum:',products[column].min())
            print('Maximum:',products[column].max())
            print('Standard Deviation:',products[column].std())
            print('Mode:', products[column].mode())
            print('---------------------------\n')
            # if column == 'TargetAmount':
            #     # Exploration of the values and distribution of the target variable
            #     sns.displot(products["TargetAmount"], kde='True',bins=50,height=6)
            #     sns.set_style('darkgrid')
            #     plt.subplots_adjust(top=0.9)
            #     plt.title("Distribution of Target Amount")
            #     plt.show()
        else:
            print(products[column].describe())
            for value, count in products[column].value_counts().items():
                print('Value -> Quantity :', value, '->', count)
    else:
        continue
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

# Partitioning the data: 70% for training and 30% for testing.
x_train, x_temp, y_train, y_temp = ms.train_test_split(x, y, test_size=0.3,train_size=0.7 , random_state=12445)
x_val, x_test, y_val, y_test = ms.train_test_split(x_temp, y_temp, test_size=0.5, random_state=12345)

print("Dimensions of Training Datasets",x_train.shape, y_train.shape)
print("Dimensions of Validation Datasets",x_val.shape, y_val.shape)
print("Dimensions of Test Datasets",x_test.shape, y_test.shape)

#endregion

#region 3. DATA MODELING

#region Run first DECISION TREE, Full Variable Model (Unpruned Decision Tree)
regressor = tr.DecisionTreeRegressor(random_state=42)
regressor.fit(x_train, y_train) # Train the model on the training data
y_pred_test1 = regressor.predict(x_test) # Make predictions on the testing data
y_pred_val1 = regressor.predict(x_val) # Make predictions on the validation data

# Plot the decision tree
# plt.figure(figsize=(12, 8))  # Adjust figure size as needed
# tr.plot_tree(regressor, feature_names=x.columns, filled=True)
# plt.show()
#endregion

#region Run second DECISION TREE, Manually Adjusted Decision Tree (Pruned with Business Insights)
parameters = {
    'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    'splitter' : ["best", "random"],
    'min_samples_split' : [2, 5, 10], # It is a percentage
    'max_depth':[1,2,3,4,5],
    'max_features' : ['sqrt','log2']
}

regressor1 = tr.DecisionTreeRegressor()
cv = ms.GridSearchCV(regressor1,param_grid=parameters, cv=5)
cv.fit(x_train, y_train, ) # Train the model on the training data
print(cv.best_params_)
print(cv.best_estimator_)
score_test2 = cv.score(x_test, y_test)
print(score_test2)

y_pred_test2 = regressor.predict(x_test) # Make predictions on the testing data
y_pred_val2 = regressor.predict(x_val) # Make predictions on the validation data

#endregion

# Create a DecisionTreeRegressor object with pruning
# regressor = DecisionTreeRegressor(
#     random_state=42,
#     ccp_alpha=0.01  # Example: Cost-complexity pruning with alpha=0.01
# )

#region Evaluate the models by comparison
# mse_test1 = metrics.mean_squared_error(y_test, y_pred_test1) # Calculate the MSE
# r2_test1 = metrics.r2_score(y_test, y_pred_test1)  # Calculate R-squared
# mae_test1 = metrics.mean_absolute_error(y_test, y_pred_test1) # Calculate MAE
# score_test1 = regressor.score(x_test, y_test) # Accuracy before pruning
#
# mse_val1 = metrics.mean_squared_error(y_val, y_pred_val1) # Calculate the MSE
# r2_val1 = metrics.r2_score(y_val, y_pred_val1)  # Calculate R-squared
# mae_val1 = metrics.mean_absolute_error(y_val, y_pred_val1) # Calculate MAE
# score_val1 = regressor.score(x_val, y_val) # Accuracy before pruning
#
#
# data = [
# {"Tree Type":"Unpruned","Adjustment":"Manual","Dataset": "Test", "MSE": mse_test1 , "R^2": r2_test1, "MAE": mae_test1, "Accuracy": score_test1},
# {"Tree Type":"Unpruned","Adjustment":"Manual","Dataset": "Validation", "MSE": mse_val1 , "R^2": r2_val1, "MAE": mae_val1, "Accuracy": score_val1 },
# ]
# print(table.tabulate(data, headers="keys", tablefmt="simple_grid"))
#endregion

#endregion