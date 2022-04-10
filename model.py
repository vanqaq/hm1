Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#model
import numpy as np

class linear(object):
    def __init__(self, input_size, output_size):
        self.w = np.random.normal(loc = 0, scale = 0.01, size = (input_size, output_size))
        self.b = np.zeros((1, output_size))
    def forward(self, x):
        return x@self.w+self.b
    def __call__(self,x):
        return self.forward(x)
    def parameters(self):
        return [self.w, self.b]
    def load_state_dict(self, param):
        self.w = param[0]
        self.b = param[1]
        
class ReLU(object):
    def __init__(self):
        pass
    def forward(self, x):
        return np.max(0, x)
    def __call__(self, x):
        return self.forward(x)
    
class twolayers_mlp(object):
    def __init__(self,input_size,hidden_size,output_size,lr,l2,milestone=500,gamma=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = linear(input_size,hidden_size)
        self.relu = ReLU()
        self.fc2 = linear(hidden_size,output_size)
        self.H = 0
        self.Z = 0
        self.K = 0
        
        self.lr = lr
        self.milestone = milestone
        self.gamma = gamma
        self.lr_ = lr
        self.l2 = l2
        
    def forward(self,x):
        y = self.fc1(x)
        self.H = y
        y = self.relu(y)
        self.Z = y
        y = self.fc2(y)
        self.K = np.exp(y)
        return y
        
    def __call__(self,x):
        return self.forward(x)
        
    def parameters(self):
        return self.fc1.parameters()+self.fc2.parameters()
        
    def load_state_dict(self,param):
        self.fc1.load_state_dict(param[:2])
        self.fc2.load_state_dict(param[2:])
        
    def backward(self,x,y):
        grad = [0]*4
        for i in range(len(x)):
            x = x[i]
            z = self.Z[i]
            k = self.K[i]
            k_diag = np.diag(k)
            h = self.H[i]
            h_diag = np.diag(np.where(h>0,1.,0.))
            y_hot = np.eye(10)[y[i]]
            e = np.ones((10,1))
            df = 1/(k@e)*e.T-1/(k@y_hot.T)*y_hot
            dk = k_diag@self.fc2.w.T
            g = []
            for j in range(len(x)):
                g.append(x[j]*h_diag)
            g = np.hstack(g)
            grad[0] += (df@dk@g).T.reshape(self.input_size,self.hidden_size)
            grad[1] += df@dk@h_diag
            g = []
            for j in range(len(z)):
                g.append(z[j]*k_diag)
            g = np.hstack(g)
            grad[2] += (g.T@df.T).reshape(self.hidden_size,self.output_size)
            grad[3] += df@k_diag
        for i in range(4):
            grad[i] = grad[i]/len(x)
            
        grad[0] = grad[0]+2*self.l2*self.fc1.w
        grad[2] = grad[2]+2*self.l2*self.fc2.w
        return grad
    
    def step(self,grad):
        self.fc1.w -= self.lr_*grad[0]
        self.fc1.b -= self.lr_*grad[1]
        self.fc2.w -= self.lr_*grad[2]
        self.fc2.b -= self.lr_*grad[3]

    def lr_decay(self,epoch):
        n = int(epoch/self.milestone)
        self.lr_ = self.lr*(self.gamma**n)
