# -*- coding:utf-8 -*-
def plotboundary(x,y,plt,clf):
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
