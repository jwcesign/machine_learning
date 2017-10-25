import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

x = [1,2,3,4,5,6,7,8]
x1 = [9,10,11,12,13,14,15,16]
y = [1,2,3,5,7,6,7,8]
y1 = [9,10,11,12,13,14,15,16]

x = np.array(x).reshape(8,1)
x1 = np.array(x1).reshape(8,1)
y = np.array(y)
y1 = np.array(y1)

num_err = []

for i in range(1,10000):
    clf = linear_model.Ridge(alpha=i)
    clf.fit(x,y)
    yp = clf.predict(x1)
    num_err.append(mean_squared_error(y1,yp))

x_alpha = np.array([i for i in range(1,10000)])
print num_err[0]
plt.plot(x_alpha,num_err)
plt.show()
