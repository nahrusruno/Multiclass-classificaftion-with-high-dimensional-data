import os
os.chdir('/Users/onursurhan/SWC')
import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
# from tensorflow.keras.layers.normalization import BatchNormalization

import pandas as pd
df=pd.read_csv('trainData.csv')
x_testtt=pd.read_csv('testData.csv')
#df.describe()
for variable in df.columns:
    
    #calculate the boundries
    lower = df[variable].quantile(0.01)
    upper = df[variable].quantile(0.95)
    
    # replacing the outliers
    df[variable] = np.where(df[variable] > upper, upper, np.where(df[variable] < lower, lower, df[variable]))

for variable in x_testtt.columns:
    
    #calculate the boundries
    lower = x_testtt[variable].quantile(0.01)
    upper = x_testtt[variable].quantile(0.95)
    
    # replacing the outliers
    x_testtt[variable] = np.where(x_testtt[variable] > upper, upper, np.where(x_testtt[variable] < lower, lower, x_testtt[variable]))









Y_train=df.iloc[:,103:104]
X_train=df.iloc[:,0:103]   
## Splitting dataset into train and test
#from sklearn.model_selection import train_test_split
#X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)

#SMOTE: Synthetic Minority Over-sampling Technique for data imbalance problem
# from imblearn.over_sampling import SMOTE
# smote = SMOTE()
# x_train, y_train = smote.fit_sample(X_train, Y_train)
x_train=X_train
y_train=Y_train
#Normalizing feature values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
normalizedData = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
x_testtt= scaler.fit_transform(x_testtt)

#One hot encoding for labels
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
Y = encoder.fit_transform(y_train)
y_train=Y
# Y_testtt=encoder.fit_transform(y_test)
# y_test=Y_testtt

## Neural Network Construction
# defining some of the hyper paramenters (these can be manipulated in order to tune the model)

#@title Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()

print("Loaded the plot_curve function.")

def create_model(my_learning_rate):
  """Create and compile a deep neural net."""
  
  # All models in this course are sequential.
  model = tf.keras.models.Sequential()

  # The features are stored in a two-dimensional 28X28 array. 
  # Flatten that two-dimensional array into a a one-dimensional 
  # 784-element array.
  model.add(tf.keras.layers.Dense(120, input_dim=103))
  model.add(layers.Activation('tanh'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.5))
            
            
  model.add(tf.keras.layers.BatchNormalization())
  # Define the first hidden layer.   
  model.add(layers.Dense(units=120,kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=1e-4), activation='relu'))
  model.add(layers.BatchNormalization())
  # Define a dropout regularization layer. 
  model.add(tf.keras.layers.Dropout(rate=0.5))

  # Define the output layer. The units parameter is set to 10 because
  # the model must choose among 10 possible output values (representing
  # the digits from 0 to 9, inclusive).
  #
  # Don't change this layer.
  model.add(tf.keras.layers.Dense(units=9, activation='softmax'))     
                           
  # Construct the layers into a model that TensorFlow can execute.  
  # Notice that the loss function for multi-class classification
  # is different than the loss function for binary classification.  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    


def train_model(model, train_features, train_label, epochs,
                batch_size, validation_split):
  """Train the model by feeding it data."""

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
  #y_pred=model.predict(x_test)
 #y_pred=model.predict(x_test,batch_size=batch_size)
  #onur=model.predict(x_testtt)
  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch. 
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist

# The following variables are the hyperparameters.
learning_rate = 0.0005   ##0.0005 iyiydi
epochs = 300
batch_size = 40
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(learning_rate)


# Train the model on the normalized training set.
epochs, hist = train_model(my_model, normalizedData, y_train, 
                           epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
#print("\n Evaluate the new model against the test set:")
#my_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)


## CROSS ENTROPY LOSS ##
# from tf.keras import losses
# scce=tf.keras.losses.SparseCategoricalCrossentropy()
# scce(y_test, y_pred).np()


sonucum=my_model.predict(x_testtt)
datam=pd.DataFrame(sonucum)
C=['c1','c2','c3','c4','c5','c6','c7','c8','c9']
datam.to_csv("sonuc_onursurhan.csv", header=C,index=False)
