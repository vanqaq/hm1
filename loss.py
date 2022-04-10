Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#loss
import numpy as np

def softmax(x):
    max_x = np.max(x,axis=1).reshape(-1,1)
    x = np.exp(x-max_x) / np.sum(np.exp(x-max_x), axis = 1, keepdims = True)
    return x

def CrossEntropyLoss(y_hat,y):
    y_hat = softmax(y_hat)
    l = 0
    for i,j in enumerate(y):
        l += -np.log(y_hat[i,j]+1e-05)
    return l/len(y)