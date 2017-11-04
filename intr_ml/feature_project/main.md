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
  * 箱形图检测（单变量）：
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
  * 残差图检测：画出预测值与观察值的差距，根据差距剔除离群值
