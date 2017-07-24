
##data preprocessing
##Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##Importing dataset
 training_set=pd.read_csv('Google_Stock_Price_Train.csv')
 training_set=training_set.iloc[:,1:2].values
                            
##Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
training_set=sc.fit_transform(training_set)

##Getting the inputs and outputs
 
X_train=training_set[0:1257]
y_train=training_set[1:1258]

##Reshaping
X_train=np.reshape(X_train,(1257,1,1))


#Building RNN
#Importing Keras libraries and packages
from keras.models import Sequential
#SEQUENTIAL TO INITIAZE THE NEURAL NETWORK
from keras.layers import Dense
#dense will create the output layer 
from keras.layers import LSTM
#best RNN as it has long memory

##Initializing the RNN
regressor=Sequential()

#Adding the input layer and the LSTM layer
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

##Adding output layer
regressor.add(Dense(units=1))

##Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

##Fitting the RNN to the training set
regressor.fit(X_train,y_train,batch_size=32,epochs=200)

#Making predictions and visualizing the results
#getting the real stock price of 2017

#importing test set
 test_set=pd.read_csv('Google_Stock_Price_Test.csv')
 real_stock_price=test_set.iloc[:,1:2].values

##getting the predicted stock price of 2017
inputs=real_stock_price
#now we need to scale them as we trained the model on the scaled data
inputs=sc.transform(inputs)
#need to reshape the data as well. predict methos id expecting 3d array.
inputs=np.reshape(inputs,(20,1,1))

predicted_stock_price=regressor.predict(inputs)
#need to inverse scale to get back to original numbers
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualising the results
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#this model can only predict the stock price for next day

#homework
#get the real stock price from 2012 to 2016
real_stock_price_train=pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train=real_stock_price_train.iloc[:,1:2].values
                                                   
#get the predicted stock price from 2012 to 2016
predicted_stock_price_train=regressor.predict(X_train)
predicted_stock_price_train=sc.inverse_transform(predicted_stock_price_train)                          

#Visualising the results
plt.plot(real_stock_price_train,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price_train,color='blue',label='Predicted Google Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

##Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))









                       