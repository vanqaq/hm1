Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#validation
import numpy as np
from loss import CrossEntropyLoss

def validation(iter_,model):
    accuracy = 0
    l = []
    for x,y in iter_:
        x = np.asarray(x)
        y = np.asarray(y)
        y_hat = model(x)
        loss = CrossEntropyLoss(y_hat,y)
        accuracy += (np.argmax(y_hat,axis=1)==y).sum()
        l.append(loss.item())
    return accuracy/len(iter_), np.mean(l) 