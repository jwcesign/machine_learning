# 机器学习
##  k近临算法
### 含义
![](http://images0.cnblogs.com/blog2015/771535/201508/041623504236939.jpg) 
---
`用官方的话来说，所谓K近邻算法，即是给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的K个实例（也就是上面所说的K个邻居）， 这K个实例的多数属于某个类，就把该输入实例分类到这个类中。`
>[相关资料](http://www.cnblogs.com/ybjourney/p/4702562.html)
##  决策树
### 含义
---
`决策树是用样本的属性作为结点，用属性的取值作为分支的树结构。 
决策树的根结点是所有样本中信息量最大的属性。树的中间结点是该结点为根的子树所包含的样本子集中信息量最大的属性。决策树的叶结点是样本的类别值。决策树是一种知识表示形式，它是对所有样本数据的高度概括决策树能准确地识别所有样本的类别，也能有效地识别新样本的类别。`
>[相关资料](http://blog.csdn.net/alvine008/article/details/37760639)


##  VC维
### 定义
---
`对于一个假设空间H，如果存在m个数据样本能够被假设空间H中的函数按所有可能的 种形式分开 ，则称假设空间H能够把m个数据样本打散（shatter）。假设空间H的VC维就是能打散的最大数据样本数目m。若对任意数目的数据样本都有函数能将它们shatter，则假设空间H的VC维为无穷大。`

![](https://pic3.zhimg.com/v2-7492d14da3e2b248e2c4971f1937ad12_b.png) 
####  不能被打散
![](https://pic4.zhimg.com/v2-64faf9d2dc907120bbc9d859b35677a3_b.png) 
####  除了这种情况都可以
---

