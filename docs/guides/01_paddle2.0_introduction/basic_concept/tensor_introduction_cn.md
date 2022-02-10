# Tensor介绍


## 概述：Tensor 的概念

飞桨（PaddlePaddle，以下简称Paddle）和其他深度学习框架一样，使用**Tensor**来表示数据，在神经网络中传递的数据均为**Tensor**。

**Tensor**可以将其理解为多维数组，其可以具有任意多的维度，不同**Tensor**可以有不同的**数据类型** (dtype) 和**形状** (shape)。

同一**Tensor**的中所有元素的数据类型均相同。如果你对 [Numpy](https://numpy.org/doc/stable/user/quickstart.html#the-basics) 熟悉，**Tensor**是类似于 **Numpy 数组（array）** 的概念。

## 一、Tensor的创建

Paddle提供了多种方式创建**Tensor**，如：指定数据列表创建、指定形状创建、指定区间创建等。

### 1.1 指定数据创建
通过给定Python列表数据，可以创建任意维度（也称为轴）的Tensor，举例如下：
```python
# 创建类似向量（vector）的一维 Tensor
import paddle # 后面的示例代码默认已导入paddle模块
ndim_1_Tensor = paddle.to_tensor([2.0, 3.0, 4.0])
print(ndim_1_Tensor)
```

```text
Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [2., 3., 4.])
```

特殊地，如果仅输入单个标量（scalar）数据（例如float/int/bool类型的单个元素），则会创建形状为[1]的**Tensor**
```python
paddle.to_tensor(2)
paddle.to_tensor([2])
```
上述两种创建方式完全一致，形状均为[1]，输出如下：
```text
Tensor(shape=[1], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [2])
```

```python
# 创建类似矩阵（matrix）的二维 Tensor
ndim_2_Tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(ndim_2_Tensor)
```
```text
Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[1., 2., 3.],
        [4., 5., 6.]])
```

```python
# 创建多维 Tensor
ndim_3_Tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(ndim_3_Tensor)
```
```text
Tensor(shape=[2, 2, 5], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [[[1 , 2 , 3 , 4 , 5 ],
         [6 , 7 , 8 , 9 , 10]],

        [[11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20]]])
```
上述不同维度的**Tensor**可以可视化的表示为：

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Tensor_2.0.png?raw=true" width="800"></center>
<br><center>图1 不同维度的Tensor可视化表示</center>

> **Tensor**必须形如矩形，也就是，在任何一个维度上，元素的数量必须**相等**，如果为以下情况将会抛出异常：
```
ndim_2_Tensor = paddle.to_tensor([[1.0, 2.0],
                                  [4.0, 5.0, 6.0]])
```

```text
ValueError:
        Faild to convert input data to a regular ndarray :
         - Usually this means the input data contains nested lists with different lengths.
```

### 1.2 指定形状创建

如果要创建一个指定形状的**Tensor**，可以使用以下API：
```python
paddle.zeros([m, n])             # 创建数据全为0，形状为[m, n]的Tensor
paddle.ones([m, n])              # 创建数据全为1，形状为[m, n]的Tensor
paddle.full([m, n], 10)          # 创建数据全为10，形状为[m, n]的Tensor
```
例如，`paddle.ones([2,3])`输出如下：
```text
Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.]])
```

### 1.3 指定区间创建

如果要在指定区间内创建**Tensor**，可以使用以下API：
```python
paddle.arange(start, end, step)  # 创建以步长step均匀分隔区间[start, end)的Tensor
paddle.linspace(start, end, num) # 创建以元素个数num均匀分隔区间[start, end)的Tensor
```
例如，`paddle.arange(start=1, end=5, step=1)`输出如下：
```text
Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [1, 2, 3, 4])
```

## 二、Tensor的属性

### 2.1 Tensor的形状

查看一个**Tensor**的形状可以通过 **Tensor.shape**，形状是 **Tensor** 的一个重要属性，以下为相关概念：

1. shape：描述了Tensor每个维度上元素的数量
2. ndim： Tensor的维度数量，例如向量的维度为1，矩阵的维度为2，Tensor可以有任意数量的维度
3. axis或者dimension：指Tensor某个特定的维度
4. size：指Tensor中全部元素的个数

创建1个四维 **Tensor**，并通过图形来直观表达以上几个概念之间的关系；
```python
ndim_4_Tensor = paddle.ones([2, 3, 4, 5])
```

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Axis_2.0.png?raw=true" width="800" ></center>
<br><center>图2 Tensor的shape、axis、dimension、ndim之间的关系</center>

```python
print("Data Type of every element:", ndim_4_Tensor.dtype)
print("Number of dimensions:", ndim_4_Tensor.ndim)
print("Shape of Tensor:", ndim_4_Tensor.shape)
print("Elements number along axis 0 of Tensor:", ndim_4_Tensor.shape[0])
print("Elements number along the last axis of Tensor:", ndim_4_Tensor.shape[-1])
```
```text
Data Type of every element: paddle.float32
Number of dimensions: 4
Shape of Tensor: [2, 3, 4, 5]
Elements number along axis 0 of Tensor: 2
Elements number along the last axis of Tensor: 5
```

重新设置**Tensor**的shape在实际编程中具有重要意义，Paddle提供了reshape接口来改变Tensor的shape：
```python
ndim_3_Tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]],
                                  [[21, 22, 23, 24, 25],
                                   [26, 27, 28, 29, 30]]])
print("the shape of ndim_3_Tensor:", ndim_3_Tensor.shape)

reshape_Tensor = paddle.reshape(ndim_3_Tensor, [2, 5, 3])
print("After reshape:", reshape_Tensor.shape)
```
```text
the shape of ndim_3_Tensor: [3, 2, 5]
After reshape: [2, 5, 3]
```

在指定新的shape时存在一些技巧：

**1.** -1 表示这个维度的值是从Tensor的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。

**2.** 0 表示实际的维数是从Tensor的对应维数中复制出来的，因此shape中0的索引值不能超过Tensor的维度。

有一些例子可以很好解释这些技巧：
```text
origin:[3, 2, 5] reshape:[3, 10]      actual: [3, 10]
origin:[3, 2, 5] reshape:[-1]         actual: [30]
origin:[3, 2, 5] reshape:[0, 5, -1]   actual: [3, 5, 2]
```

可以发现，reshape为[-1]时，会将Tensor按其在计算机上的内存分布展平为一维。
```python
print("Tensor flattened to Vector:", paddle.reshape(ndim_3_Tensor, [-1]).numpy())
```
```text
Tensor flattened to Vector: [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
```
### 2.2 Tensor的数据类型

**Tensor**的数据类型，可以通过 Tensor.dtype 来查看，dtype支持：'bool', 'float16', 'float32', 'float64', 'uint8', 'int8', 'int16', 'int32', 'int64'。

* 通过Python元素创建的Tensor，可以通过dtype来进行指定，如果未指定：

    * 对于python整型数据，则会创建int64型Tensor
    * 对于python浮点型数据，默认会创建float32型Tensor，并且可以通过set_default_type来调整浮点型数据的默认类型。

* 通过Numpy数组创建的Tensor，则与其原来的数据类型保持相同。

```python
print("Tensor dtype from Python integers:", paddle.to_tensor(1).dtype)
print("Tensor dtype from Python floating point:", paddle.to_tensor(1.0).dtype)
```
```text
Tensor dtype from Python integers: paddle.int64
Tensor dtype from Python floating point: paddle.float32
```

**Tensor**不仅支持 floats、ints 类型数据，也支持复数类型数据。如果输入为复数，则**Tensor**的dtype为 ``complex64`` 或 ``complex128`` ，其每个元素均为1个复数：

```python
ndim_2_Tensor = paddle.to_tensor([[(1+1j), (2+2j)],
                                  [(3+3j), (4+4j)]])
print(ndim_2_Tensor)  
```

```text
Tensor(shape=[2, 2], dtype=complex64, place=Place(gpu:0), stop_gradient=True,
       [[(1+1j), (2+2j)],
        [(3+3j), (4+4j)]])
```

Paddle提供了**cast**接口来改变dtype：
```python
float32_Tensor = paddle.to_tensor(1.0)

float64_Tensor = paddle.cast(float32_Tensor, dtype='float64')
print("Tensor after cast to float64:", float64_Tensor.dtype)

int64_Tensor = paddle.cast(float32_Tensor, dtype='int64')
print("Tensor after cast to int64:", int64_Tensor.dtype)
```
```text
Tensor after cast to float64: paddle.float64
Tensor after cast to int64: paddle.int64
```

### 2.3 Tensor的设备位置

初始化**Tensor**时可以通过**place**来指定其分配的设备位置，可支持的设备位置有三种：CPU/GPU/固定内存，其中固定内存也称为不可分页内存或锁页内存，其与GPU之间具有更高的读写效率，并且支持异步传输，这对网络整体性能会有进一步提升，但其缺点是分配空间过多时可能会降低主机系统的性能，因为其减少了用于存储虚拟内存数据的可分页内存。当未指定place时，Tensor默认设备位置和安装的Paddle版本一致，如安装了GPU版本的Paddle，则设备位置默认为GPU。

以下示例分别创建了CPU、GPU和固定内存上的Tensor，并通过 `Tensor.place` 查看Tensor所在的设备位置：

* **创建CPU上的Tensor**
```python
cpu_Tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
print(cpu_Tensor.place)
```

```text
Place(cpu)
```

* **创建GPU上的Tensor**
```python
gpu_Tensor = paddle.to_tensor(1, place=paddle.CUDAPlace(0))
print(gpu_Tensor.place) # 显示Tensor位于GPU设备的第 0 张显卡上
```

```text
Place(gpu:0)
```

* **创建固定内存上的Tensor**
```python
pin_memory_Tensor = paddle.to_tensor(1, place=paddle.CUDAPinnedPlace())
print(pin_memory_Tensor.place)
```

```text
Place(gpu_pinned)
```

### 2.4 Tensor的名称

Tensor的名称是其唯一的标识符，为python字符串类型，查看一个Tensor的名称可以通过Tensor.name属性。默认地，在每个Tensor创建时，Paddle会自定义一个独一无二的名称。

```python
print("Tensor name:", paddle.to_tensor(1).name)
```
```text
Tensor name: generated_tensor_0
```

## 三、Tensor的操作

### 3.1 索引和切片
您可以通过索引或切片方便地访问或修改 Tensor。Paddle 使用标准的 Python 索引规则与 Numpy 索引规则，与 [Indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings)类似。具有以下特点：

1. 基于 0-n 的下标进行索引，如果下标为负数，则从尾部开始计算
2. 通过冒号 ``:`` 分隔切片参数 ``start:stop:step`` 来进行切片操作，其中 start、stop、step 均可缺省

#### 访问 Tensor
* 针对一维 **Tensor**，则仅有单个轴上的索引或切片：
```python
ndim_1_Tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", ndim_1_Tensor.numpy())
print("First element:", ndim_1_Tensor[0].numpy())
print("Last element:", ndim_1_Tensor[-1].numpy())
print("All element:", ndim_1_Tensor[:].numpy())
print("Before 3:", ndim_1_Tensor[:3].numpy())
print("From 6 to the end:", ndim_1_Tensor[6:].numpy())
print("From 3 to 6:", ndim_1_Tensor[3:6].numpy())
print("Interval of 3:", ndim_1_Tensor[::3].numpy())
print("Reverse:", ndim_1_Tensor[::-1].numpy())
```
```text
Origin Tensor: [0 1 2 3 4 5 6 7 8])
First element: [0]
Last element: [8]
All element: [0 1 2 3 4 5 6 7 8]
Before 3: [0 1 2]
From 6 to the end: [6 7 8]
From 3 to 6: [3 4 5]
Interval of 3: [0 3 6]
Reverse: [8 7 6 5 4 3 2 1 0]
```

* 针对二维及以上的 **Tensor**，则会有多个维度上的索引或切片：
```python
ndim_2_Tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print("Origin Tensor:", ndim_2_Tensor.numpy())
print("First row:", ndim_2_Tensor[0].numpy())
print("First row:", ndim_2_Tensor[0, :].numpy())
print("First column:", ndim_2_Tensor[:, 0].numpy())
print("Last column:", ndim_2_Tensor[:, -1].numpy())
print("All element:", ndim_2_Tensor[:].numpy())
print("First row and second column:", ndim_2_Tensor[0, 1].numpy())
```
```text
Origin Tensor: [[ 0  1  2  3]
                [ 4  5  6  7]
                [ 8  9 10 11]]
First row: [0 1 2 3]
First row: [0 1 2 3]
First column: [0 4 8]
Last column: [ 3  7 11]
All element: [[ 0  1  2  3]
              [ 4  5  6  7]
              [ 8  9 10 11]]
First row and second column: [1]
```

索引或切片的第一个值对应第 0 维，第二个值对应第 1 维，以此类推，如果某个维度上未指定索引，则默认为 ``:`` 。例如：
```python
ndim_2_Tensor[1]
ndim_2_Tensor[1, :]
```
这两种操作的结果是完全相同的。

```text
Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [4, 5, 6, 7])
```

#### 修改 Tensor

> **注意：**
>
> 请慎重通过索引或切片修改 Tensor，该操作会**原地**修改该 Tensor 的数值，且原值不会被保存。如果被修改的 Tensor 参与梯度计算，将仅会使用修改后的数值，这可能会给梯度计算引入风险。Paddle 之后将会对具有风险的操作进行检测和报错。

与访问 Tensor 类似，修改 Tensor 可以在单个或多个轴上通过索引或切片操作。同时，支持将多种类型的数据赋值给该 Tensor，当前支持的数据类型有：`int`, `float`, `numpy.ndarray`, `Tensor`。

```python
import numpy as np

x = paddle.to_tensor(np.ones((2, 3)).astype(np.float32)) # [[1., 1., 1.], [1., 1., 1.]]

x[0] = 0                      # x : [[0., 0., 0.], [1., 1., 1.]]
x[0:1] = 2.1                  # x : [[2.09999990, 2.09999990, 2.09999990], [1., 1., 1.]]
x[...] = 3                    # x : [[3., 3., 3.], [3., 3., 3.]]

x[0:1] = np.array([1,2,3])    # x : [[1., 2., 3.], [3., 3., 3.]]

x[1] = paddle.ones([3])       # x : [[1., 2., 3.], [1., 1., 1.]]
```

---

同时，Paddle 还提供了丰富的 Tensor 操作的 API，包括数学运算、逻辑运算、线性代数等100余种 API，这些 API 调用有两种方法：
```python
x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float64")
y = paddle.to_tensor([[5.5, 6.6], [7.7, 8.8]], dtype="float64")

print(paddle.add(x, y), "\n") # 方法一
print(x.add(y), "\n") # 方法二
```

```text
Tensor(shape=[2, 2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       [[6.60000000 , 8.80000000 ],
        [11.        , 13.20000000]])

Tensor(shape=[2, 2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       [[6.60000000 , 8.80000000 ],
        [11.        , 13.20000000]])
```

可以看出，使用 **Tensor 类成员函数** 和 **Paddle API** 具有相同的效果，由于 **类成员函数** 操作更为方便，以下均从 **Tensor 类成员函数** 的角度，对常用 **Tensor** 操作进行介绍。

### 3.2 数学运算
```python
x.abs()                       #逐元素取绝对值
x.ceil()                      #逐元素向上取整
x.floor()                     #逐元素向下取整
x.round()                     #逐元素四舍五入
x.exp()                       #逐元素计算自然常数为底的指数
x.log()                       #逐元素计算x的自然对数
x.reciprocal()                #逐元素求倒数
x.square()                    #逐元素计算平方
x.sqrt()                      #逐元素计算平方根
x.sin()                       #逐元素计算正弦
x.cos()                       #逐元素计算余弦
x.add(y)                      #逐元素相加
x.subtract(y)                 #逐元素相减
x.multiply(y)                 #逐元素相乘
x.divide(y)                   #逐元素相除
x.mod(y)                      #逐元素相除并取余
x.pow(y)                      #逐元素幂运算
x.max()                       #指定维度上元素最大值，默认为全部维度
x.min()                       #指定维度上元素最小值，默认为全部维度
x.prod()                      #指定维度上元素累乘，默认为全部维度
x.sum()                       #指定维度上元素的和，默认为全部维度
```

Paddle对python数学运算相关的魔法函数进行了重写，例如：
```text
x + y  -> x.add(y)            #逐元素相加
x - y  -> x.subtract(y)       #逐元素相减
x * y  -> x.multiply(y)       #逐元素相乘
x / y  -> x.divide(y)         #逐元素相除
x % y  -> x.mod(y)            #逐元素相除并取余
x ** y -> x.pow(y)            #逐元素幂运算
```

### 3.3 逻辑运算
```python
x.isfinite()                  #判断Tensor中元素是否是有限的数字，即不包括inf与nan
x.equal_all(y)                #判断两个Tensor的全部元素是否相等，并返回形状为[1]的布尔类Tensor
x.equal(y)                    #判断两个Tensor的每个元素是否相等，并返回形状相同的布尔类Tensor
x.not_equal(y)                #判断两个Tensor的每个元素是否不相等
x.less_than(y)                #判断Tensor x的元素是否小于Tensor y的对应元素
x.less_equal(y)               #判断Tensor x的元素是否小于或等于Tensor y的对应元素
x.greater_than(y)             #判断Tensor x的元素是否大于Tensor y的对应元素
x.greater_equal(y)            #判断Tensor x的元素是否大于或等于Tensor y的对应元素
x.allclose(y)                 #判断Tensor x的全部元素是否与Tensor y的全部元素接近，并返回形状为[1]的布尔类Tensor
```

同样地，Paddle对python逻辑比较相关的魔法函数进行了重写，以下操作与上述结果相同。
```text
x == y  -> x.equal(y)         #判断两个Tensor的每个元素是否相等
x != y  -> x.not_equal(y)     #判断两个Tensor的每个元素是否不相等
x < y   -> x.less_than(y)     #判断Tensor x的元素是否小于Tensor y的对应元素
x <= y  -> x.less_equal(y)    #判断Tensor x的元素是否小于或等于Tensor y的对应元素
x > y   -> x.greater_than(y)  #判断Tensor x的元素是否大于Tensor y的对应元素
x >= y  -> x.greater_equal(y) #判断Tensor x的元素是否大于或等于Tensor y的对应元素
```

以下操作仅针对bool型Tensor：
```python
x.logical_and(y)              #对两个布尔类型Tensor逐元素进行逻辑与操作
x.logical_or(y)               #对两个布尔类型Tensor逐元素进行逻辑或操作
x.logical_xor(y)              #对两个布尔类型Tensor逐元素进行逻辑亦或操作
x.logical_not(y)              #对两个布尔类型Tensor逐元素进行逻辑非操作
```

### 3.4 线性代数
```python
x.t()                         #矩阵转置
x.transpose([1, 0])           #交换第 0 维与第 1 维的顺序
x.norm('fro')                 #矩阵的弗罗贝尼乌斯范数
x.dist(y, p=2)                #矩阵（x-y）的2范数
x.matmul(y)                   #矩阵乘法
```

> **注意**
>
> Paddle中API有原位（inplace）操作和非原位操作之分。原位操作即在原**Tensor**上保存操作结果，非原位（inplace）操作则不会修改原**Tensor**，而是返回一个新的**Tensor**来表示运算结果。在Paddle2.1后，部分API有对应的原位操作版本，在API后加上 `_` 表示，如`x.add(y)`是非原位操作，`x.add_(y)`为原位操作。

更多Tensor操作相关的API，请参考 [class paddle.Tensor](../../../api/paddle/Tensor_cn.html)

## 四、Tensor 与 numpy数组 相互转换
### 4.1 Tensor转换为numpy数组
通过 Tensor.numpy() 方法，将 **Tensor** 转化为 **Numpy数组**：
```python
tensor_to_convert = paddle.to_tensor([1.,2.])
tensor_to_convert.numpy()
```
```text
array([1., 2.], dtype=float32)
```
### 4.2 numpy数组转换为Tensor
通过paddle.to_tensor() 方法，将 **Numpy数组** 转化为 **Tensor**：
```python
tensor_temp = paddle.to_tensor(np.array([1.0, 2.0]))
print(tensor_temp)
```
```text
Tensor(shape=[2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       [1., 2.])
```
创建的 **Tensor** 与原 **Numpy array** 具有相同的形状与数据类型。

## 五、Tensor 的广播操作
Paddle和其他框架一样，提供的一些API支持广播(broadcasting)机制，允许在一些运算时使用不同形状的Tensor。
通常来讲，如果有一个形状较小和一个形状较大的Tensor，会希望多次使用较小的Tensor来对较大的Tensor执行一些操作，看起来像是较小形状的Tensor的形状首先被扩展到和较大形状的Tensor一致，然后做运算。值得注意的是，这期间并没有对较小形状Tensor的数据拷贝操作。

Paddle的广播机制主要遵循如下规则（参考 [Numpy 广播机制](https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting)）：

1. 每个Tensor至少为一维Tensor
2. 从后往前比较Tensor的形状，当前维度的大小要么相等，要么其中一个等于一，要么其中一个不存在

例如：

```python
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 4))
# 两个Tensor 形状一致，可以广播
z = x + y
print(z.shape)
# [2, 3, 4]

x = paddle.ones((2, 3, 1, 5))
y = paddle.ones((3, 4, 1))
# 从后向前依次比较：
# 第一次：y的维度大小是1
# 第二次：x的维度大小是1
# 第三次：x和y的维度大小相等
# 第四次：y的维度不存在
# 所以 x和y是可以广播的
z = x + y
print(z.shape)
# [2, 3, 4, 5]

# 相反
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 6))
# 此时x和y是不可广播的，因为第一次比较 4不等于6
# z = x + y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.
```

现在你知道什么情况下两个Tensor是可以广播的，两个Tensor进行广播语义后的结果Tensor的形状计算规则如下：

1. 如果两个Tensor的形状的长度不一致，那么需要在较小形状长度的矩阵向前添加1，直到两个Tensor的形状长度相等。
2. 保证两个Tensor形状相等之后，每个维度上的结果维度就是当前维度上较大的那个。

例如:

```python
x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 1))
z = x + y
print(z.shape)
# z的形状: [2,3,4]

x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 2))
# z = x + y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.
```
