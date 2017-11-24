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
    from numpy import mat,delete,array,zeros,mean
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

def normal_one_check(x,plot_state='false'):
    '''
    基于正态分布的一元离群点检测方法
    画图仅适用于二维
    '''
    import matplotlib.pyplot as plt
    feature_length = x.shape[1]
    sample_num = x.shape[0]
    out_vector = []
    for i in range(feature_length):
        u = sum(x[:,i])/sample_num
        theta_2 = sum((x[:,i]-u)**2)/sample_num
        m = x[:,i]-u
        index = m>3*theta_2
        for j in range(sample_num):
            if index[j] == True and j not in out_vector:
                out_vector.append(j)

    total = [0]*sample_num
    for i in out_vector:
        total[i] = 1
    if plot_state == 'true':
        plt.scatter(x[:,0],x[:,1],c=total,s=30,marker='+')
        plt.show()
    return out_vector

def one_normal_check(x,pro_threshold,plot_state='false'):
    '''
    基于一元正态分布的离群点检测方法
    画图适用于二维
    '''
    import matplotlib.pyplot as plt
    from numpy import mean,sqrt,exp,pi
    feature_length = x.shape[1]
    sample_num = x.shape[0]
    out_vector = []
    u_vector = mean(x,0)
    theta_2_vector = sum((x-u_vector)**2,0)/sample_num
    for i in range(sample_num):
        p = 1
        for j in range(feature_length):
            p *= exp(-((x[i,j]-u_vector[j])**2)/(2*theta_2_vector[j]))
            p /= (sqrt(2*pi)*sqrt(theta_2_vector[j]))
        if p<pro_threshold and i not in out_vector:
            out_vector.append(i)
    if plot_state == 'true':
        total = [0]*sample_num
        for i in out_vector:
            total[i] = 1
        if 0 not in total:
            plt.scatter(x[:,0],x[:,1],c='y',s=30,marker='+')
        else:
            plt.scatter(x[:,0],x[:,1],c=total,s=30,marker='+')
        plt.show()
    return out_vector

def mul_normal_check(x,pro_threshold,plot_state='false'):
    '''
    多元高斯分布的异常点检测
    画图适用于二维
    变量之间有关系时适用
    '''
    from numpy import sqrt,exp,pi,mean,cov,linalg,mat
    import matplotlib.pyplot as plt
    c = cov(x.T)
    u = mean(x,0)
    feature_length = x.shape[1]
    sample_num = x.shape[0]
    out_vector = []
    for i in range(sample_num):
        p = 1
        p *= exp(-(1/2.0)*(x[i]-u)*mat(c).I*mat((x[i]-u)).T)
        p /= (2*pi)**(feature_length/2.0)*(linalg.det(c)**0.5)
        if p<pro_threshold and i not in out_vector:
            out_vector.append(i)
    if plot_state == 'true':
        total = [0]*sample_num
        for i in out_vector:
            total[i] = 1
        plt.scatter(x[:,0],x[:,1],c=total,s=30,marker='+')
        plt.show()
    return out_vector

def mad_check(x,dis_threshold,plot_state='false'):
    '''
    用 Mahalanobis(马氏) 距离检测多元离群点
    notes:当你的数据表现出非线性关系关系时，你可要谨慎使用该方法了，马氏距离仅仅把他们作为线性关系处理。
    可以把此距离作为异常点检测的一句
    '''
    from numpy import mat,sqrt,mean,linalg,cov
    import matplotlib.pyplot as plt
    u = mean(x,0)
    c = cov(x.T)
    sample_num = x.shape[0]
    out_vector = []
    for i in range(sample_num):
        mad = sqrt(mat(x[i]-u)*mat(c).I*mat(x[i]-u).T)
        print mad
        if mad > dis_threshold:
            out_vector.append(i)
    if plot_state == 'true':
        total = [0]*sample_num
        for i in out_vector:
            total[i] = 1
        plt.scatter(x[:,0],x[:,1],c=total,s=30,marker='+')
        plt.show()
    return out_vector

def x2_check(x,threshold,plot_state='false'):
    from numpy import mean
    u = mean(x,0)
    sample_num = x.shape[0]
    out_vector = []
    for i in range(sample_num):
        x2 = sum(((x[i]-u)**2)/abs(u))
        if x2 > threshold:
            out_vector.append(i)
    if plot_state == 'true':
        import matplotlib.pyplot as plt
        total = [0]*sample_num
        for i in out_vector:
            total[i] = 1
        plt.scatter(x[:,0],x[:,1],c=total,s=30,marker='+')
        plt.show()
    return out_vector

def isof_check(x,plot_state='false',con=0.01):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=con)
    clf.fit(x)
    sample_num = x.shape[0]
    total = [0]*sample_num
    out_vector = []
    py = clf.predict(x)
    for i in range(sample_num):
        if py[i] == -1:
            total[i] = 1
            out_vector.append(i)
    if plot_state == 'true':
        import matplotlib.pyplot as plt
        plt.scatter(x[:,0],x[:,1],c=total,s=30,marker='+')
        plt.show()
    return out_vector




from sklearn import preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
####LOF start
def distance_euclidean(x):
    rs = x.shape[0]
    cs = x.shape[1]
    dis = {}
    for i in range(rs):
        x_tm = np.delete(x,i,0)
        x_now = x[i]
        dis_total = np.sum((x_tm-x_now)**2,1)**0.5
        dis[i] = {}
        index = 0
        for j in range(dis_total.shape[0]):
            if dis_total[j] in dis[i]:
                dis[i][dis_total[j]].append(index+(index>=i))
            else:
                add_one = []
                add_one.append(index+(index>=i))
                dis[i][dis_total[j]] = add_one
            index += 1
    e_dis = {}
    for i in dis:
        e_dis[i] = sorted(dis[i].items())
    return e_dis

def k_distance(x,k):
    k_tem = {}
    for i in x:
        k_tem[i] = x[i][0:k]
    k_dis = {}
    for i in k_tem:
        k_dis[i] = k_tem[i][k-1][0]
    return k_dis

def k_reach_distance(neighbors,k_dis,k):
    k_r_dis = {}
    for i in neighbors:
        k_r_dis[i] = {}
        for j in neighbors[i]:
            if j[0]>k_dis[i]:
                for m in j[1]:
                    k_r_dis[i][m] = j[0]
            else:
                for m in j[1]:
                    k_r_dis[i][m] = k_dis[i]
    return k_r_dis

def k_neighbors(x,k):
    k_n = {}
    for i in x:
        k_n[i]  = []
        for j in x[i][0:k]:
            k_n[i].append(j[1][0])
    return k_n

def lrd(k_n,k_reach_dis):
    each_lrd = {}
    for i in k_n:
        each_lrd[i] = 0
        for j in k_n[i]:
            each_lrd[i] += k_reach_dis[j][i]
        each_lrd[i] = len(k_n[i])/each_lrd[i]
    return each_lrd
def lof(k_n,each_lrd):
    each_lof = {}
    for i in k_n:
        each_lof[i] = 0
        for j in k_n[i]:
            each_lof[i] += each_lrd[j]
        each_lof[i] = (each_lof[i]/len(k_n[i]))/each_lrd[i]
    return each_lof

def local_outlier_factor_self(min_pts,x):
    neighbors_array = distance_euclidean(x)
    k_n = k_neighbors(neighbors_array,min_pts)
    k_dis = k_distance(neighbors_array,min_pts)
    k_reach_dis = k_reach_distance(neighbors_array,k_dis,min_pts)
    each_lrd = lrd(k_n,k_reach_dis)
    return lof(k_n,each_lrd)

class LOF:
    """
    Use lof to judge the degree as a outlier
    """
    def __init__(self,plot_state=False,degree=2.0,min_pts=5):
        self.degree = degree
        self.plot_state = plot_state
        self.min_pts = min_pts

    def fit(self,x):
        x = np.array(x)
        self.x = pp.minmax_scale(x)

    def judge_self(self):
        result = local_outlier_factor_self(self.min_pts,self.x)
        return result
        arr_re = []
        for i in result:
            if result[i] > self.degree:
                arr_re.append(-1)
            else:
                arr_re.append(1)
        return np.array(arr_re)

    def predict():
        pass

###LOF end
