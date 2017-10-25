# -*- coding:utf-8 -*-
def plot_boundary(x,y,plt,clf):
    '''
    * 函数说明
    * 对二维有效
    * 如果需要请降维
    * x为特征值，y为label值，clf为svm函数（已fit）
    '''
    import numpy as np
    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 1000)
    yy = np.linspace(ylim[0], ylim[1], 1000)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,linestyles=['--', '-', '--'])
    plt.show()

def plot_colormap(x,y,plt,clf,classses):
    '''
    画决策边界图，不同的类，不同的颜色, classses为类的数目，也就是y中不同数的个数
    适用于二维，其他维数不行
    '''
    from matplotlib.colors import ListedColormap
    import numpy as np
    # step size in the mesh
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z)
    plt.scatter(x[:,0],x[:,1],c=y,edgecolor='k')
    plt.xlim()
    plt.ylim()
    plt.show()

def plot_3d_scatter(x,y,z,color_map):
    ''''
    画3维散点图，并根据类标记颜色
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(111,projection='3d')
    ax.scatter(x,y,z,c=color_map,s=1)
    plt.show()
