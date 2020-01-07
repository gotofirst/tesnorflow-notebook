# TensorFlow入门之基础介绍
## 一、Tensorflow的安装与配置
这部分CSDN上有很多教程，跟着教程一步步来就行了。
## 二、TensorFlow总体介绍
### 2.1 张量（Tensor）
张量是所有深度学习框架中最核心的组件，因为后续的所有运算和优化算法都是基于张量进行的。几何代数中定义的张量是基于向量和矩阵的推广，通俗一点理解的话，我们可以将标量视为零阶张量，矢量视为一阶张量，那么矩阵就是二阶张量。
有了张量对象之后，下面一步就是一系列针对这一对象的数学运算和处理过程。所谓的“学习”就是不断纠正神经网络的实际输出结果和预期结果之间误差的过程。这里的一系列操作包含的范围很宽，可以是简单的矩阵乘法，也可以是卷积、池化和LSTM等稍复杂的运算。
在Tensorflow中，张量可以被简单地理解为多维数组。张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。输出张量输出的是对结果的一个引用：

```
Tensor("add:0",shape=(2,),dtype=float32)
```
一个张量中主要保存了三个属性：名字（name）、维度（shape）、类型（type）。名字属性是一个张量的唯一标识符，它同时也给出了这个张量是如何计算出来的。张量的命名可以通过“node:src_output”的形式给出，其中node为节点的名称，src_output表示当前张量来自节点的第几个输出。（编号从0开始）。第二个属性是维度，shape=(2,)表示张量是一个一维数组，长度为2.第三个属性是类型，每一个张量会有一个唯一的类型，TensorFlow会对参与运算的所有张量进行类型的检查 ， 当发现类型不匹配时会报错。如果不指定类型， TensorFlow 会给出默认的类型，比如不带小数点的数会被默认为 int32，带小数点的会默认为 float32 。因为使用默认类型有可能会导致潜在的类型不匹配问题，所以一般建议通过指定 dtype 来明确指出变量或者常
量的类型 。TensorFlow 支持 14 种不同的类型，主要包括了实数（ tf.float32 、 tf.float64 ）、整数（ tf.int8 、 tf.intl 6 、 tf.int32 、 tf.int64 、 tf.uint8 ）、布尔型 ( tf.bool) 和复数（ tf.complex64 、tf.complex128 ） 。
和 TensorFlow 的计算模型相比， TensorFlow 的数据模型相对比较简单 。 张量使用主要可以总结为两大类 。第一类用途是对中间计算结果的引用 。 当一个计算包含很多中间结果时，使用张量可以大大提高代码的可读性。使用张量的第二类情况是当计算图构造完成之后，张量可以用来获得计算结果，也就是得到真实的数字。
### 2.2 计算图（Graph）
有了张量和基于张量的各种操作之后，下一步就是将各种操作整合起来，输出我们需要的结果。
但不幸的是，随着操作种类和数量的增多，有可能引发各种意想不到的问题，包括多个操作之间应该并行还是顺次执行，如何协同各种不同的底层设备，以及如何避免各种类型的冗余操作等等。这些问题有可能拉低整个深度学习网络的运行效率或者引入不必要的Bug，而计算图正是为解决这一问题产生的。
计算图带来的另一个好处是让模型训练阶段的梯度计算变得模块化且更为便捷，也就是自动微分法。
将待处理数据转换为张量，针对张量施加各种需要的操作，通过自动微分对模型展开训练，然后得到输出结果开始测试。
TensorFlow是一个通过计算图的形式来表达计算的编程系统。TensorFlow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。
TensorFlow程序一般有两个阶段。第一个阶段：定义计算图中所有的计算；第二个阶段：执行计算。
在TensorFlow程序中，系统会维护一个默认的计算图，通过tf.get_default_graph函数可以获取当前默认的计算图。除了使用默认的计算图，TensorFlow支持通过tf.Graph函数来生成新的计算图。不同的计算图上的张量和运算都不会共享。
TensorFlow中的计算图不仅仅可以用来隔离张量和计算，它还提供了管理张量和计算的机制。计算图可以通过tf.Graph.device函数来指定运行计算的设备。
指定、调用GPU/CPU:
/cpu:0机器的CPU；/gpu:0机器的第一个GPU；/gpu:1机器的第二个GPU

```
With tf.Session() as sess:
	With tf.dervice(“/gpu:1”):
		………
```
### 2.3 会话（session）
会话拥有并管理 TensorFlow 程序运行时的所有资源。所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄漏的问题。 TensorFlow 中使用会话的模式一般有两种，第一种模式需要明确调用会
话生成函数和关闭会话函数。
为了解决异常退出时资源释放的问题， TensorFlow 可以通过 Python 的上下文管理器来使用会话。通过 Python 上下文管理器的机制，只要将所有的计算放在 “ with”的内部就可以 。 当上下文管理器退出时候会自动释放所有资源。这样既解决了因为异常退出时资源释放的问题，同时也解决了忘记调用 Session.close 函数而产生的资源泄漏。具体使用见后面。
在交互式环境下（比如 Python 脚本或者 Jupyter 的编辑器下），通过设置默认会话的方式来获取张量的取值更加方便 。 所以 TensorFlow 提供了 一种在交互式环境下直接构建默认会话的函数。这个函数就是tf.lnteractiveSession。使用这个函数会自动将生成的会话注册为默认会话。

```
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
```
但是无论使用哪种方法都可以通过 ConfigProto Protocol BufferCD来配置需要生成的会话 。

```
config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
```
通过 ConfigProto 可以配置类似并行的线程数、 GPU 分配策略、运算超时时间等参数。第一个是allow_soft_placement，通过将 allow_soft_placement 参数设为 True ， 当某些运算无法被当前 GPU 支持时，可 以自动调整到 CPU 上，而不是报错。类似地，通过将这个参数设置为 True，可以让程序在拥有不同数量的 GPU 机器上顺利运行。
### 2.4 TensorFlow总体
用张量tensor表示数据；计算图graph表示计算任务；在会话session中的上下文context中执行图；通过变量Variable维护状态；通过feed和fetch可以任意的操作（arbitrary operation）、赋值、获取数据。
Tensorflow是一个编程系统，使用图（ graphs）来表示计算任务，图（ graphs）中的节点称之为op（ operation），一个op获得0个或多个Tensor，执行计算，产生0个或多个Tensor。 Tensor 看作是一个 n 维的数组或列表。图必须在会话（ Session）里被启动。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019100719014245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
两个数据进行op产生一个新的数据（tensor），然后这个tensor继续接着下面的op计算；
右边的：w先与x相乘，结果再与b进行加法操作，接着进行ReLU。
## 三、代码实例
### 3.1 创建图，运行图
TensorFlow中有一个默认的图，我们把数据，计算节点加入其中就可以计算了。

```
import tensorflow as tf
#创建图、启动图
m1 = tf.constant([[3,3]])     #创建一个一行两列的向量常量
m2 = tf.constant([[2],[3]])   #创建一个两行一列的向量常量
product = tf.matmul(m1,m2)    #创建一个乘法op
print(product)

sess = tf.Session()           #创建一个会话,启动默认图
result = sess.run(product)    #调用sess的run方法执行矩阵乘法op
print(result)
sess.close()                  #关闭会话

#常用写法：
with tf.Session() as sess:
    result = sess.run(product)  # 调用sess的run方法执行矩阵乘法op
    print(result)
```
首先创建数据，变量或者是常量；然后创建op，加法或者乘法等计算，然后创建会话启动默认图，在会话中运行op操作，最后关闭会话。
一般按后面的写法编程，方便简洁。
输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191007191443195.png)
前面一个输出输出的是一个tensor，因为只是定义了一个op，并未在会话中进行，所以输出是一个tensor。
### 3.2 变量的使用
创建变量然后赋值，然后变量都需要初始化。

```
x = tf.Variable([1,2])        #创建一个变量
a = tf.constant([3,3])
sub = tf.subtract(x,a)        #创建一个减法op
add = tf.add(x,sub)             #创建一个加法op

init = tf.global_variables_initializer()   #初始化全局变量
with tf.Session() as sess:
    sess.run(init)            #会话中运行变量初始化操作
    print(sess.run(sub))
    print(sess.run(add))
```
首先创建变量；然后创建op，然后需要进行变量初始化，接着创建会话启动默认图，在会话中运行op操作。
输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191007191417122.png)
下面是一个变量自增的操作：

```
state = tf.Variable(0,name='counter')     #创建一个变量并赋值起名字
new_value = tf.add(state,1)
update = tf.assign(state,new_value)       #赋值操作，把后面那个的值赋给前面的
init = tf.global_variables_initializer()   #初始化全局变量

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
```
输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019100719130484.png)
### 3.3 Fetch的使用
Fetch的作用主要是：在会话中同时运行多个op。

```
#Fetch：在会话中同时运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(6.0)

add = tf.add(input1,input2)
mul = tf.multiply(add,input3)

with tf.Session() as sess:
    result = sess.run([add,mul])  #同时运行多个op
    print(result)
```
在中括号中是运行的多个op。
运行结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191007191800626.png)
2+3=5；5*6=30
### 3.4 Feed的使用

```
#Feed:
input4 = tf.placeholder(tf.float32)    #创建一个32位浮点型的占位符
input5 = tf.placeholder(tf.float32)    #创建一个32位浮点型的占位符
output = tf.multiply(input4,input5)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input4:[7.],input5:[4.]}))  #feed的数据以字典的形式传入,即分别给两个占位符赋值
```
输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191007193915217.png)
### 3.5 一个实例

```
#一个简单的案例
import numpy as np
x_data = np.random.rand(100)   #随机生成100个点
y_data = x_data*0.1 + 0.2

#构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))    #计算误差的平方，再求它的平均值
#调用梯度下降法来进行训练
optimizer = tf.train.GradientDescentOptimizer(0.2)  #学习率是0.2
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #迭代训练
    for step in range(201):
        sess.run(train)
        #20次打印一下参数
        if step % 20 == 0:
            print(step,sess.run([k,b]))
```
输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/201910071940413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhaGEwODI1,size_16,color_FFFFFF,t_70)
也就是最后k约为0.1，b为0.2的时候效果最好。

