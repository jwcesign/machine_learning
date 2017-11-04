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

def box_check(x,time=0.7413,delete_state='false',plot_state='false'):
    '''
    time为1.5时为中度异常，为3时为极度异常。但可根据具体情况调节该参数
    '''
    import matplotlib.pyplot as plt
    from numpy import mat,array,delete,sort
    feature_length = x.shape[1]
    sample_num = x.shape[0]
    A = (sample_num+1)/4.0
    B = 3*(sample_num+1)/4.0
    total_return = []
    for i in range(feature_length):
        x_tm = sort(x[:,i])
        q1 = x_tm[int(A)]+(A-int(A))*(x_tm[int(A)+2]-x_tm[int(A)+1])
        q3 = x_tm[int(B)]+(B-int(B))*(x_tm[int(B)+2]-x_tm[int(B)+1])
        iqr = (q3-q1)*time
        out_vector_big = x[:,i]>q3+iqr
        out_vector_small = x[:,i]<q1-iqr
        for j in range(sample_num):
            if out_vector_small[j] == True or out_vector_big[j] == True:
                if j not in total_return:
                    total_return.append(j)
    if plot_state == 'true':
        plt.boxplot(x,sym='r+')
        plt.show()
    if delete_state == 'true':
        x = delete(x,total_return,0)
        return x
    return total_return

def cook_dis(x,y,plot_state='false',time=4,delete_state='false'):
    # 适用于回归模型
    from numpy import mat,delete,array
    from sklearn import linear_model
    clf = linear_model.LinearRegression()
    clf.fit(x,y)
    py = clf.predict(x)

    sample_num = x.shape[0]
    feature_length = x.shape[1]
    dis_vector = zeros(sample_num)
    diff = mat(y-py)
    for i in range(sample_num):
        x_tm = delete(x,i,0)
        y_tm = delete(y,i,0)
        clf.fit(x_tm,y_tm)
        py_tm = clf.predict(x)
        s_2 = diff*diff.T/(sample_num-feature_length)
        dis = sum((py-py_tm)**2)/(feature_length*s_2)
        dis_vector[i] = dis
    if plot_state == 'true':
        import matplotlib.pyplot as plt
        plt.stem(range(sample_num),dis_vector)
        mean_dis = mean(dis_vector)
        plt.plot(range(sample_num),[mean_dis*time]*sample_num)
        plt.show()
    if delete_state == 'true':
        mean_delete = mean(dis_vector)*time
        index_delete = dis_vector>mean_delete
        delete_vector = []
        for i in range(sample_num):
            if index_delete[i] == True:
                delete_vector.append(i)
        x = delete(x,delete_vector,0)
        y = delete(y,delete_vector,0)
        return x,y
    return dis_vector

def diff_value(x,y,delete_state='false',threshold=3,plot_state='false'):
    '''
    画标准残差值
    '''
    from numpy import sqrt,delete
    from sklearn import linear_model
    import matplotlib.pyplot as plt
    clf = linear_model.LinearRegression()
    clf.fit(x,y)
    py = clf.predict(x)
    dis = abs(y-py)/sqrt(sum((y-py)**2)/y.shape[0])
    if plot_state == 'true':
        plt.stem(range(y.shape[0]),dis)
        plt.show()
    if delete_state == 'true':
        tm = dis > threshold
        delete_vector = []
        for i in range(y.shape[0]):
            if tm[i] == True:
                delete_vector.append(i)
        x = delete(x,delete_vector,0)
        y = delete(y,delete_vector,0)
        return x,y
