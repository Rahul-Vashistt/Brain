import pandas as pd
import numpy as np
from brainml.utils import __lr_schedule

def Batch_GD(X,y,learning_rate,epochs, tol):
    theta = np.ones(X.shape[1]) #(p+1,)
    n = X.shape[0] #number of samples

    for epoch in range(epochs):
        try:
            lr = __lr_schedule(epoch) if learning_rate is None else float(learning_rate)
        except:
            raise TypeError(f'Expected Integers or float But got {learning_rate}')

        y_pred = np.dot(X,theta)
        residual = y_pred - y
        Gradient_thetha = 2/n * np.dot(X.T,residual)

        theta -= lr * Gradient_thetha

        # Convergence check
        if np.linalg.norm(Gradient_thetha) < tol:
            break
    return theta

    
