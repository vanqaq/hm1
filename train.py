Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
from mlxtend.data import loadlocal_mnist
from Mydataloader import MydataLoader
from model import twolayers_mlp
from loss import SoftmaxLossLayer
from validation import validation
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 

x_train, y_train = loadlocal_mnist(
            images_path=r'C:\Users\13693\Desktop\mlnn\hm1\train-images.idx3-ubyte', 
            labels_path=r'C:\Users\13693\Desktop\mlnn\hm1\train-labels.idx1-ubyte')
x_test, y_test = loadlocal_mnist(
            images_path=r'C:\Users\13693\Desktop\mlnn\hm1\t10k-images.idx3-ubyte', 
            labels_path=r'C:\Users\13693\Desktop\mlnn\hm1\t10k-labels.idx1-ubyte')

def normalize(x):
    m = np.mean(x,axis=0,keepdims=True)
    var = np.var(x,axis=0,keepdims=True)
    return (x-m)/np.sqrt(var+1e-05)
x_train = normalize(x_train)
x_test = normalize(x_test)

input_size = 28*28
hidden_size_list = [256,128]
output_size = 10
lr_list = [0.1, 0.01]
l2_list = [1e-04,5e-04]

epoch = 1
batch_size = 64

result = []

for hidden_size in hidden_size_list:
    for lr in lr_list:
        for l2 in l2_list:
            print(' ')
            print('hidden_size:{},lr:{},l2:{}'.format(hidden_size,lr,l2))
            
            train_iter = Mydataloader(x_train,y_train,batch_size)
            test_iter = Mydataloader(x_test,y_test,batch_size) 
            network = twolayers_mlp(input_size,hidden_size,output_size,lr,l2)
            
            train_loss=[]
            test_loss=[]
            accuracy = []
    
            for iteration,data in enumerate(train_iter):
                iteration += 1
                x,y = data
                y_hat = network(x)
                l = network.softmax.get_loss(y)
                network.backward()
                network.update(lr)
                
                train_loss.append(l.item())
                acc, test_l = validation(test_iter, network)
                test_loss.append(test_l.item())
                accuracy.append(acc.item())
                network.lr_decay(iteration)
                
                if iteration%100 == 0:
                    print('iteration:{},loss:{},accuracy:{}'.format(iteration,test_l,acc))
            
            network.save_model(r'C:\Users\13693\Desktop\mlnn\hm1\mlp2-{}hidden-{}lr-{}l2.npy'.format(hidden_size, lr, l2)
            num = list(range(len(train_loss)))
            plt.figure(figsize=(6,6),dpi=100)
            plt.plot(num,train_loss,label='trainset loss')
            plt.plot(num,test_loss,label = 'testset loss')
            plt.xlabel('iteration')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(r'C:\Users\13693\Desktop\mlnn\hm1\loss_{}_{}_{}.png'.format(hidden_size,lr,l2))
            
            plt.figure(figsize=(6,6),dpi=100)
            plt.plot(num,accuracy)
            plt.xlabel('iteration')
            plt.ylabel('testset accuracy')
            plt.savefig(r'C:\Users\13693\Desktop\mlnn\hm1\acc_{}_{}_{}.png'.format(hidden_size,lr,l2))
            
            result.append([hidden_size,lr,l2,acc])
            
df = pd.DataFrame(result,columns=['hidden_size','lr','l2','accuracy'])
df.to_csv(r'C:\Users\13693\Desktop\mlnn\hm1\result.csv',index=False)
