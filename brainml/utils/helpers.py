from brainml.method import optimizers
import numpy as np
import pandas as pd

def resolve_optimizer(name: str) -> str:
    name = name.lower()
    
    for key, aliases in optimizers.items():
        if name in [alias.lower() for alias in aliases]:
            return key
          
    raise ValueError(f"❌ Unknown optimizer '{name}'. Valid options are: {sum(optimizers.values(), [])}") 


def __lr_schedule(t: float) -> float:
        t0,t1 = 5,50
        return t0/(t+t1)


def is_standardized(X, tol=1e-3):
     mean = np.mean(X, axis=0)
     std = np.std(X, axis=0)

     return np.all(np.abs(mean) < tol) and np.all(np.abs(std - 1) < tol)

def is_minmax_scaled(X, tol=1e-3):
     return np.all(np.abs(np.min(X,axis=0) < tol)) and np.all(np.abs(np.max(X,axis=0) - 1) < tol)

     