#特征工程
## 总图
![img](http://images2015.cnblogs.com/blog/927391/201604/927391-20160430145122660-830141495.jpg)

## 1.处理异常样本
1. 数据预处理：判断变量是否符合高斯分布,数据如果不是高斯分布，尽量转化为高斯分布：数据整体偏小，可以求ln(x)或者x^a,0<a<1;数据整体偏大,可以求e^x或者x^a,a>1。   
~~~python
#import ****
plt.hist(x[:,i],100) #100为分成几份
plt.show()
~~~
![img](https://raw.githubusercontent.com/jwcesign/machine_learning/master/pic_save/1.png)
2. 缺失值处理
  * 删除
  * 如果缺失值占总体比例非常小，那么直接填入相关的平均值。
  * 如果不算大也不算小时，可以利用没有缺失值的数据构建模型，然后用来预测缺失值。
  * KNN寻找最进点，然后进行估计。
  * 如果缺失值所占比例较大，则直接把缺失值当作一种特殊情况，另取一个值填入缺失值。
3. 离群值检测
  * 箱形图检测(单变量）：
  ~~~python
  # 箱形图画法
  import pandas as pd
  df = pa.DataFrame(x)
  df.columns = ['A','B',...]
  df.boxplot()
  plt.show(sym='r+') #离群点表示
  ~~~
  * 箱形图原理：
    * 四分位距离：将数据从小到大排序，有N个值，设A=(N+1)/4,B=3(N+1)/4,则：下四分位为 Q1=X{[A]}+(A-[A])(X{[A]+2}-X{[A]+1}),上四分位为Q3=X{[B]}+(B-[B])(X{[B]+2}-X{[B]+1})。四分位距离=Q3-Q1.
    * 设定一个最小估计值和最大估计值，如果超出这个范围，则认为为异常值，最小估计值为：Q1-k(Q3-Q1),最大估计值为：Q3+k(Q3-Q1),k一般取1.5（中度异常）或者3（极度异常）    
![img](https://raw.githubusercontent.com/jwcesign/machine_learning/master/pic_save/2.png)
  * 残差图检测：画出预测值与观察值的差距，根据差距剔除离群值,一般大于2或3的为离群值
![img](https://raw.githubusercontent.com/jwcesign/machine_learning/master/pic_save/3.png)
4. 强影响点检测：
  * cook's D:适用与回归模型，原理见代码：
  ![img](http://www.zhihu.com/equation?tex=D%7B_i%7D%3D%5Cfrac%7B%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Cleft%28+%5Chat%7BY%7D_%7Bj%7D+-+%5Chat%7BY%7D_%7Bj+%5Cleft%28i+%5Cright%29%7D+%5Cright%29%5E%7B2%7D%7D%7Bp+%5Ctimes+MSE%7D)
5. 其他：[link](blog.csdn.net/mr_tyting/article/details/77371157)
### 补充
#### LOF: local outlier factor
> 可以通过密度检测来给定一个点为异常点的程度，可以解决分布不均的问题。

![img](http://img.blog.csdn.net/20160618150545625)
#### DBSCAN: 可用与异常点检测，但不是很好
> [详解](http://blog.csdn.net/itplus/article/details/10088625)

## 2.数据不均衡
> 如果不均衡比例超过4:1，分类器就会偏向于大的类别。

### 扩充数据

### 对数据进行重采样
* 过采样：对小类的数据样本进行过采样来增加小类的数据样本个数，即采样的个数大于该类样本的个数。
* 欠采样：对大类的数据样本进行欠采样来减少大类的数据样本个数，即采样的个数少于该类样本的个数。
* 合成样本：SMOTE，Border-smote算法。
* 使用带惩罚的模型：可以为每类给定一个权重。比如通过给观测值少的类较大的代价。常见的比如penalized-SVM或者penalized-LDA。
* easyEnsemble: 通过多次欠采样解决问题。
* BalanceCasecade: 跟easyEnsemble原理差不多。
* Note: 一般情况下欠采样效果会更好。

### 补充
#### Adaboost
![img](https://raw.githubusercontent.com/jwcesign/machine_learning/master/pic_save/1335428125_1739.png)

[详细](http://www.360doc.com/content/14/1109/12/20290918_423780183.shtml)
