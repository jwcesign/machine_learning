# Sklearn学习
***
`提示：用haroopad软件打开可看到数学公式`
## 常用命令
~~~python
from sklearn import some_model
import matplotlib.pyplot as plt
plt.scatter(x,y) # 画点
plt.plot(x,reg.predict(x)) # 画线
# Pipline的使用, ***先后关系***
from skearn.pipeline import Pipeline
model = Pipeline([('poly', PolynomialFeatures(degress=3)),('linear',LinearRegression())])
x = np.arange(5)
y = 3 - 2 * x + x ** 2 - x ** 3
model = model.fit(x[:, np.newaxis], y)
model.named_steps['linear'].coef_
~~~

## 关于画图
~~~python
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplo3d import Axes3D
# 画三维散点图
figure = plt.figure()
ax = Axes3D(figure)
ax.scatter(x,y,z,c='r',marker='*')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
# 画三维线
ax.plot3D(x,y,z)
plt.show()
# 画面
x,y = np.meshgrid(x,y)
z = x+y
ax.plot_surface(x,y,z,cmap='rainbow')
plt.show()
~~~

## 线性回归：
### 普通最小二乘法
* 对各个变量都**公平**，其系数估计依赖模型各项相互独立
* 原理
$$
\underset{w}{min\,} {|| X w - y||_2}^2
$$
* 代码
~~~python
from sklearn import linear_model
reg = linear_model.LinearRegression()
x = [...]
y = [...]
reg.fit()
~~~

### 岭回归：
* 当变量之间**有关系**时用
* 原理: *aplha*越大，收缩率越大，那么系数对于共线性的鲁棒性更强
$$
\underset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}
$$
* 代码
~~~python
from sklearn import linear_model
# reg = liear_model.RidgeCV(alphas = alphas)
# reg.fit(x,y)
reg = linear_model.Ridge(alpha = m)
reg.fit(x,y)
~~~

### Lasso回归：
* **变量相关且噪声较多**时用
* 描述：The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is dependent.
* 原理：
$$
\underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}
$$
* 代码
~~~python
from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])
## Result: array([ 0.8])
~~~

### Elastic Net(弹性网络)
* 当多个特征和另一个特征相关的时候弹性网络非常有用
* 原理
$$
\underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}
$$

### Least Angle Regression(最小角度回归)
* 最小角回归是针对高维数据的回归算法
* 优势:当 p >> n 时计算是非常高效的。（比如当维数远大于点数）
* 代码
~~~python
from sklearn import linear_model
reg = linear_model.Lars(n_nonzero_coefs=1)
reg.fit(x,y)
~~~

### LARS Lasso
* 是一个使用LARS算法实现的lasso模型。
* 代码
~~~python
from sklearn import linear_model
clf = linear_model.LassoLars(alpha = 0.1)
clf.fit(x,y)
~~~

### 贝叶斯岭回归
* Bayesian Ridge Regression 对于病态问题更具有鲁棒性。
* 代码
~~~python
from sklearn import linear_model
x = [...]
y = [...]
clf = linear_mmodel.BayesianRidge()
clf.fit()
~~~

### Automatic Relevance Determination - ARD
* 和贝叶斯岭回归非常相似，主要针对稀疏权重
* 代码
~~~python
from sklearn import linear_model
clf  = linear_model.ARDRegression()
clf.fit(x,y)
~~~

### Logistic Regression(逻辑回归)
* 二分类问题
* 代码
~~~python
from sklearn.linear_model import LogisticRegression
# The larger m is, the model has more freedom
reg = LogisticRegression(C=m)
~~~

### Stochastic Gradient Descent - SGD
* 随机梯度下降(SGD)是一种快速拟合线性模型非常有效的方式,尤其当样本数量非常大的时候非常有用。
* 对于设置参数 `loss="log"` ,SGDClassifier 拟合了一个逻辑回归模型，而设置参数 `loss="hinge"` ,该类会拟合一个线性SVM
* 代码
~~~python
from sklearn.lieanr_model import SGDClassifier
clf = SGDClassifier(loss='log') # or loss='hinge'
clf.fit(x,y)
~~~

### Perceptron(感知机)
* 特点
 * 它不需要学习率
 * 不需要正则化(罚项)
 * 只会在判错情况下更新模型。
* 代码
~~~python
from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(x,y)
~~~

### Passive Aggressive Algorithms
* 算法和感知机非常相似，并不需要学习率。但是和感知机不同的是，这些算法都包含有一个正则化参数 C 。
* 对于分类问题, **PassiveAggressiveClassifier** 可以通过设置 `loss='hinge'` (PA-I) 或者 `loss='squared_hinge'` (PA-II)来处理。 对于回归问题, **PassiveAggressiveRegressor** 可以通过设置 `loss='epsilon_insensitive'` (PA-I) 或者 `loss='squared_epsilon_insensitive'` (PA-II)来处理。
* 代码
~~~python
from sklearn.linear_model import PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier() # C=...
clf.fit(x,y)
~~~

### 鲁棒（稳健）回归
* Robust regression(稳健回归) 主要思路是对异常值十分敏感的经典最小二乘回归目标函数的修改。 它主要用来拟合含异常数据(要么是异常数据,要么是模型错误) 的回归模型。

### Theil-Sen regression
*  It is thus robust to multivariate outliers. Note however that the robustness of the estimator decreases quickly with the dimensionality of the problem. It looses its robustness properties and becomes no better than an ordinary least squares in high dimension.
* 经实验，相比于OSL，模型更鲁棒，低维
### Huber Regression
*  It differs from TheilSenRegressor and RANSACRegressor because it does not ignore the effect of the outliers but gives a lesser weight to them.

### Polynomial regression: extending linear models with basis functions
* 实质是线性回归
~~~python
//使用
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(10,1)
y = x[:]**2+4-2*x[:]
poly = PolynomialFeatures(degree = 2)
x1 = poly.fit_transform(x)
lin = linear_model.LinearRegression()
lin.fit(x1)
plt.scatter(x,y)
plt.plot(x,predict(x1),color="blue")
plt.show()
~~~
### KernelRidge
* Kernel ridge regression (KRR) combines ridge regression (linear least squares with l2-norm regularization) with the kernel trick.
* 代码
~~~python
from sklearn.kernel_ridge import KernelRidge
clf = KernelRidge(..parameter..)
clf.fit(x,y)
~~~

### 实践
* 对于岭回归
~~~python
'''
对于实验结果，alpha越大, mean_squared_error越大
'''
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
~~~


### 总结
* 各个模型的用处不一样，如对噪声的反映
* 各个模型的参数会对模型产生影响，用help(model_name)查看参数解释
### 补充
#### 数据降维
* ***PAC(principal component analysis)***
  * PCA的本质就是找一些投影方向，使得数据在这些投影方向上的方差最大，而且这些投影方向是相互正交的。***(忽略标签的影响,希望降维后的数据能够保持最多的信息)***
  ~~~python
  from sklearn.decomposition import PCA
  # 参数为要降到的维数, 要保留多少的相同性
  pca = PCA(n_components=2)
  x = pca.fit(x).transform(x)
  ~~~
* ***LDA(Linear Discriminant Analysis)***
  * 相比于PCA，***加入了标签对降维的影响***
  * LDA不适合对非高斯分布样本进行降维(**希望数据在降维后能够很容易地被区分开来**)。
  ~~~python
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  lda = LinearDiscriminantAnalysis((n_components=2))
  x_r2 = lda.fit(x,y).transform(x)
  ~~~


## Support Vector Machine
* SVMs are a set of supervised learning methods used for classification, regression and outliers detection
* The advantage:
  1. Effective in high dimensional spaces.
  2. Still effective in cases where number of dimensional is greater than the number of samples.
  3. 核函数可选:

  ![img](http://scikit-learn.org/stable/_images/sphx_glr_plot_iris_0012.png)
  4. 代码
  ~~~python
  # 对于核函数的选择，主要是linear和RBF(高斯核心函数)
  # SVC，模型构建可选OVR,OVO等
  from sklearn import svm
  clf = svm.SVC()
  clf.fit(x,y)
  clf.predict(x_test)
  # LinearSVC,采用OVR的模型构建,这个只有linear核心函数
  clf = scm.LinearSVC()
  lin_clf.fit(x,y)
  # 画图技巧
  # x=[2d],会把类区分出来
  plt.scatter(x[:,0],x[:,1],c=y)

  ~~~
* 对于类别不平衡的数据，可以通过设置SVM的class_weight={label: weight}来改善模型
~~~python
x = [...]
y = [0,0,0,0,0,0,0,0,0,0,1,1,1]
# 还有sample_weight可以设置
clf = svm.SVC(linear='some_one',class_weight={1:10})
clf.fit(x,y)
~~~

## Support Vector Regression
* svm.SVR()
* 代码
~~~python
from sklearn.svm import SVR
# 选择kernel，设置相关参数
clf = SVR(kernel='kernel',...)
clf.fit(x,y)
~~~
* 图示

![img](http://scikit-learn.org/stable/_images/sphx_glr_plot_svm_regression_001.png)

* 从图示可以看出：***The model produced by support vector classification (as described above) depends only on a subset of the training data***

## novelty detection
* 用于非监督学习
* OneClassSVM:
~~~python

~~~

### 特征选择（百分比，结合label选择）
* 代码
~~~python
from sklearn import feature_selection
tran = feature_selection.SelectPercentile(feature_selection.f_classif)
# 设置percentile以设置特征选取百分比
x = tran.fit(x,y).transform(x)
~~~
