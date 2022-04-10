Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#model
import numpy as np

class linear(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.normal(loc = 0, scale = 0.01, size = (self.input_size, self.output_size))
        self.b = np.zeros((1, self.output_size))
        
    def forward(self, x):
        self.x = x
        self.output = np.matmul(self.x,self.w)+self.b
        
    def backward(self,top_diff):
        self.d_w=np.dot(self.x.T,top_diff)
        self.d_b=np.sum(top_diff,axis=0)
        bottom_diff=np.dot(top_diff,self.w.T)
        return bottom_diff        
        
    def update_param(self,lr):
        self.weight=self.weight-lr*self.d_w
        self.bias=self.bias-lr*self.d_b
        
    def load_param(self,w,b):
        self.w=w
        self.b=b
        
    def save_param(self):
        return self.w,self.b
        
class ReLU(object):
    def __init__(self):
        pass
    def forward(self, x):
        self.x = x
        return np.maximum(0, self.x)
    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.x < 0] = 0
        return bottom_diff
    
class twolayers_mlp(object):
    def __init__(self,input_size,hidden_size,output_size,lr,l2,milestone=500,gamma=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.milestone=milestone
        self.gamma=gamma
        self.lr=lr
        self.lr_=lr
        self.l2=l2
        self.fc1 = linear(self.input_size,self.hidden_size)
        self.relu = ReLU()
        self.fc2 = linear(self.hidden_size,self.output_size)
        self.softmax = SoftmaxLossLayer()
        self.update_layer = [self.fc1, self.fc2]
        
    def forward(self,x):
        y = self.fc1.forward(x)
        y = self.relu.forward(y)
        y = self.fc2.forward(y)
        prob = sefl.softmax.forward(y)
        return prob
        
    def __call__(self,x):
        return self.forward(x)
    
    def init_model(self):
        for layer in self.update_layer:
            layer.init_param()
 
    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        np.save(param_dir, params)
        
    def load_model(self, param_dir):
        params = np.load(param_dir,allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
      
    def backward(self):
        dloss = self.softmax.backward()
        dh2 = self.fc2.backward(dloss)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)
        
        def update(self, lr):
            for layer in self.update_layer:
                layer.update_param(lr)
        
    def lr_decay(self,epoch):
        n = int(epoch/self.milestone)
        self.lr_ = self.lr*(self.gamma**n)
