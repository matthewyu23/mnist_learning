from keras.datasets import mnist
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = np.array(train_X)[:1000] / 255
train_y = np.array(train_y)[:1000]
test_X = np.array(test_X)[:1000] / 255
test_y = np.array(test_y)[:1000]

np.random.seed(1)


relu = lambda x:(x>=0) * x
relu2 = lambda x: x>=0

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

alpha = 0.005
weights_0_1 = 0.1 * (2*np.random.random((784, 40)) - 1)
weights_1_2 = 0.1 * (2*np.random.random((40, 10)) - 1) 



for iteration in range(350): 
    layer_2_error = 0
    for i in range(len(train_X)): 
        layer_0 = train_X[i]
        layer_0 = np.array([layer_0.flatten()])
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)
        
        

        answer = np.zeros(10)
        answer[train_y[i]] = 1
        answer = [answer]
        layer_2_error += np.sum((answer - layer_2) ** 2)
   

        
        layer_2_delta = np.array(answer - layer_2)
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2(layer_1)


        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)



    

    print(layer_2_error)


score = 0 
for i in range(len(test_X)): 
    layer_0 = np.array([test_X[i].flatten()])
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = relu(np.dot(layer_1, weights_1_2))
    


    prediction = max_elem(layer_2)
    answer = test_y[i]
    

    if (prediction == answer): 
        score = score + 1
    
print(score / len(test_y))