import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# ############################# Importing the dataset ################################################
dataset = pd.read_csv(r'D:\DEEPA\Data Science\Spyder codes\salary_data.csv')
print(dataset)

# ############################# Exploring data using functions #######################################
dataset.shape # 30 rows, 2 columns
dataset.columns #Years of experience, Salary
dataset.info() 
dataset.describe()
dataset.isnull().sum() # No null values
dataset.head()

# ############################# Exploring data using graphs (Data Visualization) #####################
# 1) Box plot - To know max, min, median, 25 percentile, 75 percentile
plt.boxplot(dataset["YearsExperience"])
plt.title("YearsExperience")
plt.show()
plt.boxplot(dataset["Salary"])
plt.title("Salary")
plt.show()

# 2) Histogram - To know distribution of columns
plt.hist(dataset["YearsExperience"])
plt.title("YearsExperience")
plt.show()
plt.hist(dataset["Salary"])
plt.title("Salary")
plt.show()

# 3) Scater plot - To know relation between two
plt.scatter(dataset["YearsExperience"], dataset["Salary"])
plt.title("Relation")
plt.show()
#Observation - Appears to be linear relation

# 4) Scatter Matrix - Best to know relation
scatter_matrix(dataset,figsize=(8,8))
plt.show()


# ############################# To know co-relation ##################################################
corr_matrix = dataset.corr()
print(corr_matrix)

# ############################# Splitting the data into train and test dataset ######################
X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
y = dataset.iloc[:, 1].values #get array of dataset in  1st column

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# ############################# Applying Machine Learning Algorithm ################################

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualizing the Training set results

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting the result of 5 Years Experience
y_pred = regressor.predict(np.array(5).reshape(-1,1))
y_pred

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

#Checking the accuracy
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#R-squared score

print('R-squared score:', regressor.score(X_test, y_test))  