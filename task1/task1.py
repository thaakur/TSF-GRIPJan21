#!/usr/bin/env python
# coding: utf-8

# ### Data Science And Business Analytics - GRIPJAN 2021

# ### By : Avinash Kumar

# ### TASK 1 : Prediction using supervised learning
# ### GRIP @ THE SPARKS FOUNDATION

# In this task , I'll predict the percentage of marks that a student is expected to score based on data of how much he have studied

# This is a sample linear regression task as it involves only two variables

# ### Importing required libraries

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ### Reading data from source i.e. url/remote link

# In[4]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data is imported")
s_data.head(25)


# ### Input data visualization

# In[5]:


#plotting the distribution of scores
s_data.plot(x = 'Hours' , y = 'Scores' , style = 'o')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()


# From graph we can assume a positive linear relation between the number of hours studied and percentage of score

# ### Data Preprocessing
# Dividing the data into Attributes(input) and Labels(output)

# In[6]:


data_x = s_data.iloc[:, :-1].values
data_y = s_data.iloc[:, 1].values


# ### Model Training

# Splitting data into training and testing sets and training the model after that.

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train.reshape(-1,1), y_train) 

print("Data Model Trained.")


# ### Plotting the regression line

# Now as our model is trained now, it's time to test it and visualize the best-fit line of regression.

# In[9]:


#plotting the regression line
line = regressor.coef_*data_x+regressor.intercept_

#plotting for the test data
plt.scatter(data_x,data_y)
plt.plot(data_x, line, color='black')
plt.show()


# ### Testing the Model for prediction

# As the model is trained now , let's test it by making some predictions
# 
# For this we will use the test-set data

# In[10]:


#testing data
print(X_test)
#model prediction
pred_y = regressor.predict(X_test)


# ### Comparing the results

# Comparing the actual result with the predicted result

# In[11]:


df = pd.DataFrame({'Actual' : y_test, 'Predicted': pred_y})
df.head()


# In[12]:


#comparing training score with test score
print("Trained Score : ",regressor.score(X_train,y_train))
print("Tested Score : ",regressor.score(X_test,y_test))


# In[15]:


# Plotting the Bar graph to depict the difference between the actual and predicted value

df.plot(kind='bar',figsize=(6,6))
plt.grid(which='major', linewidth='0.5', color='green')
plt.grid(which='minor', linewidth='0.5', color='yellow')
plt.show()


# In[16]:


# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### Conclusion
# 
# ### From the above results we verify and say that if a student studied for 9.25 hours then he certainly would score 93.69 marks.

# In[ ]:





# In[ ]:




