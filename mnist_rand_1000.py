import mnist
import pickle
import numpy as np

x,t,_,__ = mnist.load()

mask = np.random.choice(x.shape[0], 1000, replace=False)

x_ = x[mask]
t_ = t[mask]

mnist1000 = {"training_images": x_,
             "training_labels": t_}

with open('mnist1000.pkl', 'wb') as f:
    pickle.dump(mnist1000, f)
