from keras.datasets import mnist
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = np.array(train_X) / 255
train_y = np.array(train_y)
test_X = np.array(test_X) / 255
test_y = np.array(test_y)

np.random.seed(1)


def relu(x):
    answer = []
    for i in x[0]: 
        if i > 0: 
            answer.append(i)
        else: 
            answer.append(0)
    return np.array([answer])

def relu2(x): 
    answer = []
    for i in x[0]: 
        if i > 0: 
            answer.append(1)
        else: 
            answer.append(0)
    return np.array([answer])

def max_elem(x): 
    arr = x[0]

    index = 0
    m = arr[0]

    for i in range(len(arr)): 
        if arr[i] > m: 
            index = i
            m = arr[i]

    return index

np.random.seed(1)

alpha = 0.0001
hidden_layer_size = 64
weights_0_1 = 2*np.random.random((784, hidden_layer_size)) - 1 
weights_1_2 = 2*np.random.random((hidden_layer_size, hidden_layer_size)) - 1 
weights_2_3 = 2*np.random.random((hidden_layer_size, 10)) - 1 



for iteration in range(30): 
    layer_3_error = 0
    for i in range(len(train_X)): 
        layer_0 = train_X[i]
        layer_0 = np.array([layer_0.flatten()])
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = relu(np.dot(layer_1, weights_1_2))
        layer_3 = np.dot(layer_2, weights_2_3)
        
        

        answer = np.zeros(10)
        answer[train_y[i]] = 1
        layer_3_error += np.sum((layer_3 - answer) ** 2)
        

        
        layer_3_delta = (layer_3 - answer) 
        layer_2_delta = np.dot(layer_3_delta, weights_2_3.T) * relu2(layer_2)
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2(layer_1)

        weights_2_3 -= alpha * layer_2.T.dot(layer_3_delta)
        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)




        
    print(layer_3_error)


score = 0 
for i in range(len(test_X)): 
    layer_0 = np.array([test_X[i].flatten()])
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = relu(np.dot(layer_1, weights_1_2))
    layer_3 = np.dot(layer_2, weights_2_3)


    prediction = max_elem(layer_3)
    answer = test_y[i]
    

    if (prediction == answer): 
        score = score + 1
    
print(score / len(test_y))