# -*-coding:utf-8-*-

from sklearn import preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt

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
        arr_re = []
        for i in result:
            if result[i] > self.degree:
                arr_re.append(-1)
            else:
                arr_re.append(1)
        return np.array(arr_re)

    def predict():
        pass
