import pandas as pd
import numpy as np
from brainml.utils import __lr_schedule

def Mini_Batch_GD(X,y,learning_rate,epochs,batch_size,n_batches, tol):
    theta = np.ones(X.shape[1]) # (p+1,)
    n = X.shape[0] # number of samples

     # Validate batch_size / n_batches logic
    if (batch_size is None and n_batches is None) or (batch_size is not None and n_batches is not None):
        raise ValueError("Provide exactly one of 'batch_size' or 'n_batches'.")

    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        n_batches = int(np.ceil(n / batch_size))
    else:
        if not isinstance(n_batches, int):
            raise TypeError("n_batches must be an integer.")
        batch_size = int(np.ceil(n / n_batches))
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for batch in range(0,n,batch_size):
            start = batch
            end = min((batch + batch_size),n)

            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            if learning_rate is None:
                lr = __lr_schedule(epoch)
            else:
                if not isinstance(learning_rate,(float,int)):
                    raise TypeError(f'Expected float but got {learning_rate}')
                lr = float(learning_rate)

            y_pred = np.dot(x_batch,theta)
            residual = y_pred - y_batch
            Gradient_thetha = (2/batch_size) * (x_batch.T @ residual)

            theta -= lr * Gradient_thetha
    
        # Convergence check
        if np.linalg.norm(Gradient_thetha) < tol:
            break
    return theta