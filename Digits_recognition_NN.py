"""
Title: Kaggle Digit Recognition data (using scikit-Learn's MLP)

@author: Vishnuvardhan Janapati
"""

#import matplotlib.pyplot as plt                             # enable when platting image
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import time

start_time=time.time()  # check computational time for this classification

# loading training data and scaling
ImageId=pd.read_csv('sample_submission.csv')
ImageId=ImageId['ImageId']
digits=pd.read_csv('train.csv')
y=digits['label']
X=digits.drop(['label'],1)
scaler=preprocessing.StandardScaler()
X=scaler.fit_transform(X)



## loading test data and scaling
digits_test=pd.read_csv('test.csv')
digits_test=scaler.transform(digits_test)


#   Model selection
mlp=MLPClassifier(hidden_layer_sizes=(50,50,20),max_iter=50,tol=1e-5, alpha=1e-5,solver='sgd',verbose=10,random_state=1,learning_rate_init=0.01)

## ------------ Test model parameters and optimize for better performance before testing with unlabel data 
#X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#mlp.fit(X_train,y_train)
#
#print("Accuracy of classifying training set is " + str(mlp.score(X_train,y_train)*100) + " percent")
#print("Accuracy of classifying test set is " + str(mlp.score(X_test,y_test)*100) + " percent")
#

## for comparison
#for i in range(1,20):
##    plt.imshow(np.reshape(digits_test[i,:],(28,28)))
#    if i==1:
#        print("         Actual digits          Predicted digit       ")
#    print("              ", str(y_test.values[i]) ,"                    " , str(mlp.predict(np.reshape(X_test[i,:],(1,-1)))))
    



### ------------ Train with given label data (X,y), and 
mlp.fit(X,y)
print("Accuracy of classifying training set is " + str(mlp.score(X,y)*100) + " percent")

# plot image of digit for false-prediction cases
count=0
for i in range(len(X)):
    if(mlp.predict(reshape(X[i],(1,-1)))[0]!=y[i]):
        print(i)
        count+=1
        print('Actual digit is ',y[i], 'and is predicted as ',mlp.predict(reshape(X[i],(1,-1))))
        plt.imshow(np.reshape(X[i],(28,28)))
        plt.show()
print('Number of false predictions: ',count)
       
####  test model accuracy with unlabel data 
predictions=mlp.predict(scaler.transform(digits_test))
#
### writing output to a csv file
submission=pd.DataFrame({'ImageId':ImageId,'Label':predictions})
submission.to_csv('submission.csv',index=False)

print('time elapsed :',time.time()-start_time)
