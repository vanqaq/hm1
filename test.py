Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
from mlxtend.data import loadlocal_mnist
from Mydataloader import MydataLoader
from model import twolayers_mlp 

X_test, y_test = loadlocal_mnist(
            images_path=r'C:\Users\13693\Desktop\mlnn\hm1\t10k-images-idx3-ubyte', 
            labels_path=r'C:\Users\13693\Desktop\mlnn\hm1\t10k-labels-idx1-ubyte')
            
def normalize(x):
    m = np.mean(x,axis=0,keepdims=True)
    var = np.var(x,axis=0,keepdims=True)
    return (x-m)/np.sqrt(var+1e-05)
x_test = normalize(x_test)

input_size = 28*28
hidden_size = 256
output_size = 10
batch_size = 32

test_iter = MydataLoader(x_test,y_test,batch_size) 
network = twolayers_mlp(input_size,hidden_size,output_size,lr = 0,l2 = 0)
param = np.load(r'C:\Users\13693\Desktop\mlnn\hm1\param.npy', allow_pickle=True)
network.load_state_dict(param)

accuracy = 0
for x,y in test_iter:
    x = np.asarray(x)
    y = np.asarray(y)
    y_hat = network(x)
    accuracy += (np.argmax(y_hat,axis=1)==y).sum()
print('accuracy:',accuracy/len(test_iter))