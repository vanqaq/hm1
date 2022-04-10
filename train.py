Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
from mlxtend.data import loadlocal_mnist
from Mydataloader import MydataLoader
from model import twolayers_mlp
from loss import CrossEntropyLoss
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
hidden_size_list = [512,256,128]
output_size = 10
lr_list = [0.2,0.1]
l2_list = [1e-04,5e-04]

batch_size = 32

d = []

for hidden_size in hidden__size_list:
    for lr in lr_list:
        for l2 in l2_list:
            print(' ')
            print('hidden_size:{},lr:{},l2:{}'.format(hidden_size,lr,l2))
            
            train_iter = Mydataloader(x_train,y_train,batch_size)
            test_iter = Mydataloader(x_test,y_test,batch_size) 
            network = twolayers_mlp(input_size,hidden_size,output_size,lr,l2)
            
            ce_train = []
            ce_test = []
            accuracy = []
    
            for iteration,data in enumerate(train_iter):
                iteration += 1
                x,y = data
                x = np.asarray(x)
                y = np.asarray(y)
                y_hat = network(x)
                l = CrossEntropyLoss(y_hat,y)
                grad = network.backward(x,y)
                network.step(grad)
                
                ce_train.append(l.item())
                acc, ce = validation(test_iter, network)
                ce_test.append(ce.item())
                accuracy.append(acc.item())
                network.lr_decay(iteration)
                
                if iteration%10 == 0:
                    print('iteration:{},loss:{},accuracy:{}'.format(iteration,ce,acc))
            
            np.save(r'C:\Users\13693\Desktop\mlnn\hm1\param_'+str(hidden_size)+'_'+str(lr)+'_'+str(l2)+'.npy',network.parameters())
            num = list(range(len(ce_train)))
            plt.figure(figsize=(6,6),dpi=100)
            plt.plot(num,ce_train,label='trainset loss')
            plt.plot(num,ce_test,label = 'testset loss')
            plt.xlabel('iteration')
            plt.ylabel('CrossEntropyLoss')
            plt.legend()
            plt.savefig(r'C:\Users\13693\Desktop\mlnn\hm1\loss_'+str(hidden_size)+'_'+str(lr)+'_'+str(l2)+'.png')
            
            plt.figure(figsize=(6,6),dpi=100)
            plt.plot(num,accuracy)
            plt.xlabel('iteration')
            plt.ylabel('testset accuracy')
            plt.savefig(r'C:\Users\13693\Desktop\mlnn\hm1\accuracy_'+str(hidden_size)+'_'+str(lr)+'_'+str(l2)+'.png')
            
            d.append([hidden_size,lr,l2,acc])
            
df = pd.DataFrame(d,columns=['hidden_size','lr','l2','accuracy'])
df.to_csv(r'C:\Users\13693\Desktop\mlnn\hm1\result.csv',index=False)
