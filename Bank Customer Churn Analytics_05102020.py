#Part1

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(10)


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X= dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 3 Classes Encoding
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])], remainder = 'passthrough')
X= np.array(ct.fit_transform(X))

# 2 Classes Encoding
labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])

X=np.around(X.astype(np.double),1)     #delete the float and do the round

X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Part2

#Import
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialising the ANN

classifier = Sequential()


#Input Layer and Hidden layer 


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))



#Second Hidden Layer 

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))



#Output Layer 

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#Compling the ANN (how to optimize weight)

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


classifier.save('model')


#Fitting the ANN to the Trainning set 

'classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)'

train_history=classifier.fit(X_test, y_test, validation_split=0.1, batch_size = 10, epochs = 100)




# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix-
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test,y_pred)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Part 4 - Train model v.s. Validation model

# Diagram


def show_train_history (train_history,train,validation):
    
    
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    

show_train_history(train_history,'accuracy','val_accuracy')

show_train_history(train_history,'loss','val_loss')



# Confusion_Matrix

import confusion_matrix


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

display_labels=['Stay','Leave']

disp = ConfusionMatrixDisplay(cm, display_labels)

disp = disp.plot(include_values=bool)

plt.show()







'''
# Part 5 - Adjust Parameter

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [3, 5],
              'optimizer': ['adam', 'rmsprop']}
              
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 2)


grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
'''