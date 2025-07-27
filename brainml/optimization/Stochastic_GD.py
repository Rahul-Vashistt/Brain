import pandas as pd
import numpy as np
from brainml.utils import __lr_schedule

def Stochastic_GD(X,y,learning_rate,epochs, tol):
    theta = np.ones(X.shape[1]) # (p+1,)
    n = X.shape[0] # number of samples

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for step in range(n):
            x_i = X_shuffled[step]
            y_i = y_shuffled[step]

            try:
                lr = __lr_schedule(epoch) if learning_rate is None else float(learning_rate)
            except:
                raise TypeError(f'Expected Integers or float But got {learning_rate}')

            y_pred = theta * x_i
            residual = y_pred - y_i
            Gradient_thetha = 2 * residual * x_i

            theta -= lr * Gradient_thetha
    
        # Convergence check
        if np.linalg.norm(Gradient_thetha) < tol:
            break
    return theta