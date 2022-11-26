# Ex-6-Handwritten Digit Recognition using MLP
## Aim:
       To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
/*Describe here*/


## Algorithm :


## Program:
```
Developed by: N.Niharika. 
RegisterNumber:  212221240031.
```
```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

```
```
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

```
```
Y_train
```
```

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```
```
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
 ```
 ```
 W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
 ```
 ```
 def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    ```
    ```
    test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
```
```
 dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```

## Output :

![203900084-784dec43-a206-4697-aa07-9140e30928f2](https://user-images.githubusercontent.com/94165377/204083403-cd067fd9-6630-4b0f-8d26-56cf6434f7da.jpg)



![203900106-96070289-5ec9-4e4e-a063-3feae26ed45c](https://user-images.githubusercontent.com/94165377/204083404-269e76c8-73c7-477d-bbf5-469d94425a52.jpg)

![203900132-e7a5d6f0-292e-4ec1-88a7-fa85a397b73f](https://user-images.githubusercontent.com/94165377/204083405-8d5bb281-6577-4a08-b427-de7214a024b4.jpg)


![203900151-e483c6a7-9ef1-43f5-95d6-27a4f809906a](https://user-images.githubusercontent.com/94165377/204083410-730a933b-ffd3-4b7f-bd70-f217cff6b4e3.jpg)
![203900172-b75f78dd-f44f-4bfe-8dff-53df9c2d243e](https://user-images.githubusercontent.com/94165377/204083412-0ff30682-004d-4ed0-9309-bd6564cf5906.jpg)
![203900190-7a0a5632-2190-47b6-a008-c9123bf4fa73](https://user-images.githubusercontent.com/94165377/204083417-386c31cc-7f85-4398-b3af-891b7e25696c.jpg)

![203900200-2ce99bec-d506-4bf1-9df8-fccd0e50ef9f](https://user-images.githubusercontent.com/94165377/204083428-71044a03-ff45-4596-a19c-182f55b05193.jpg)

![203900214-ee19f8c7-f7ea-44be-9529-a8d2ba5cccd8](https://user-images.githubusercontent.com/94165377/204083441-e3b1fad0-4074-4682-869e-b030214cc9c1.jpg)

![203900278-679d938d-bc3a-46c5-9a9c-ba5be13754de](https://user-images.githubusercontent.com/94165377/204083444-8729e058-f9d6-4569-b6f0-37b4baa30685.jpg)



## Result:
Thus The Implementation of Handwritten Digit Recognition using MLP Is Executed Successfully.
