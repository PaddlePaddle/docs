# Tensor概念介绍

飞桨（PaddlePaddle，以下简称Paddle）和其他深度学习框架一样，使用**Tensor**来表示数据，在神经网络中传递的数据均为**Tensor**。

**Tensor**可以将其理解为多维数组，其可以具有任意多的维度，不同**Tensor**可以有不同的**数据类型** (dtype) 和**形状** (shape)。

同一**Tensor**的中所有元素的dtype均相同。如果你对 [Numpy](https://www.paddlepaddle.org.cn/tutorials/projectdetail/590690) 熟悉，**Tensor**是类似于 **Numpy array** 的概念。

## Tensor的创建

首先，创建一个 **Tensor** , 并用 **ndim** 表示 **Tensor** 维度的数量:

### 1. 创建类似于vector的**1-D Tensor**，其 ndim 为1

```python
# 可通过dtype来指定Tensor数据类型，否则会创建float32类型的Tensor
ndim_1_tensor = paddle.to_tensor([2.0, 3.0, 4.0], dtype='float64')
print(ndim_1_tensor)
```

```text
Tensor(shape=[3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
       [2., 3., 4.])
```

特殊地，如果仅输入单个scalar类型数据（例如float/int/bool类型的单个元素），则会创建shape为[1]的**Tensor**
```python
paddle.to_tensor(2)
paddle.to_tensor([2])
```
上述两种创建方式完全一致，shape均为[1]，输出如下：
```text
Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [2])
```

### 2. 创建类似于matrix的**2-D Tensor**，其 ndim 为2
```python
ndim_2_tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(ndim_2_tensor)
```
```text
Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [[1., 2., 3.],
        [4., 5., 6.]])
```

### 3. 同样地，还可以创建 ndim 为3、4...N等更复杂的多维Tensor
```
# Tensor可以有任意数量的轴（也称为维度）
ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(ndim_3_tensor)
```
```text
Tensor(shape=[2, 2, 5], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [[[1, 2, 3, 4, 5],
         [ 6,  7,  8,  9, 10]],

        [[11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20]]])
```
上述不同ndim的**Tensor**可以可视化的表示为：

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Tensor_2.0.png?raw=true" width="800"></center>
<br><center>图1 不同ndim的Tensor可视化表示</center>


你可以通过 Tensor.numpy() 方法方便地将 **Tensor** 转化为 **Numpy array**：
```python
ndim_2_tensor.numpy()
```
```text
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
```

**Tensor**不仅支持 floats、ints 类型数据，也支持 complex numbers数据：

```python
ndim_2_complex_tensor = paddle.to_tensor([[1+1j, 2+2j],
                                          [3+3j, 4+4j]])
```

如果输入为复数数据，则**Tensor**的dtype为 ``complex64`` 或 ``complex128`` ，其每个元素均为1个复数：
```text
Tensor(shape=[2, 2], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
       [[(1+1j), (2+2j)],
        [(3+3j), (4+4j)]])
```

**Tensor**必须形状规则，类似于“矩形”的概念，也就是，沿任何一个轴（也称作维度）上，元素的数量都是相等的，如果为以下情况：
```
ndim_2_tensor = paddle.to_tensor([[1.0, 2.0],
                                  [4.0, 5.0, 6.0]])
```
该情况下将会抛出异常：
```text
ValueError:
    Faild to convert input data to a regular ndarray :
     - Usually this means the input data contains nested lists with different lengths.
```

上面介绍了通过Python数据来创建**Tensor**的方法，你也可以通过 **Numpy array** 来创建**Tensor**：
```python
ndim_1_tensor = paddle.to_tensor(numpy.array([1.0, 2.0]))

ndim_2_tensor = paddle.to_tensor(numpy.array([[1.0, 2.0],
                                              [3.0, 4.0]]))

ndim_3_tensor = paddle.to_tensor(numpy.random.rand(3, 2))
```
创建的 **Tensor** 与原 **Numpy array** 具有相同的 shape 与 dtype。

如果要创建一个指定shape的**Tensor**，Paddle也提供了一些API：
```text
paddle.zeros([m, n])             # 创建数据全为0，shape为[m, n]的Tensor
paddle.ones([m, n])              # 创建数据全为1，shape为[m, n]的Tensor
paddle.full([m, n], 10)          # 创建数据全为10，shape为[m, n]的Tensor
paddle.arange(start, end, step)  # 创建从start到end，步长为step的Tensor
paddle.linspace(start, end, num) # 创建从start到end，元素个数固定为num的Tensor
```

## Tensor的shape

### 基本概念
查看一个**Tensor**的形状可以通过 **Tensor.shape**，shape是 **Tensor** 的一个重要属性，以下为相关概念：

1. shape：描述了tensor的每个维度上元素的数量
2. ndim： tensor的维度数量，例如vector的 ndim 为1，matrix的 ndim 为2.
3. axis或者dimension：指tensor某个特定的维度
4. size：指tensor中全部元素的个数

创建1个4-D **Tensor**，并通过图形来直观表达以上几个概念之间的关系；
```python
ndim_4_tensor = paddle.ones([2, 3, 4, 5])
```

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Axis_2.0.png?raw=true" width="800" ></center>
<br><center>图2 Tensor的shape、axis、dimension、ndim之间的关系</center>

```python
print("Data Type of every element:", ndim_4_tensor.dtype)
print("Number of dimensions:", ndim_4_tensor.ndim)
print("Shape of tensor:", ndim_4_tensor.shape)
print("Elements number along axis 0 of tensor:", ndim_4_tensor.shape[0])
print("Elements number along the last axis of tensor:", ndim_4_tensor.shape[-1])
```
```text
Data Type of every element: VarType.FP32
Number of dimensions: 4
Shape of tensor: [2, 3, 4, 5]
Elements number along axis 0 of tensor: 2
Elements number along the last axis of tensor: 5
```


### 对shape进行操作

重新定义**Tensor**的shape在实际编程中具有重要意义。
```python
ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]],
                                  [[21, 22, 23, 24, 25],
                                   [26, 27, 28, 29, 30]]])
print("the shape of ndim_3_tensor:", ndim_3_tensor.shape)
```
```text
the shape of ndim_3_tensor: [3, 2, 5]
```

Paddle提供了reshape接口来改变Tensor的shape：
```python
ndim_3_tensor = paddle.reshape(ndim_3_tensor, [2, 5, 3])
print("After reshape:", ndim_3_tensor.shape)
```
```text
After reshape: [2, 5, 3]
```

在指定新的shape时存在一些技巧：

**1.** -1 表示这个维度的值是从Tensor的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
**2.** 0 表示实际的维数是从Tensor的对应维数中复制出来的，因此shape中0的索引值不能超过x的维度。

有一些例子可以很好解释这些技巧：
```text
origin:[3, 2, 5] reshape:[3, 10]      actual: [3, 10]
origin:[3, 2, 5] reshape:[-1]         actual: [30]
origin:[3, 2, 5] reshape:[0, 5, -1]   actual: [3, 5, 2]
```

可以发现，reshape为[-1]时，会将tensor按其在计算机上的内存分布展平为1-D Tensor。
```python
print("Tensor flattened to Vector:", paddle.reshape(ndim_3_tensor, [-1]).numpy())
```
```text
Tensor flattened to Vector: [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
```

## Tensor其他属性
### Tensor的dtype

**Tensor**的数据类型，可以通过 Tensor.dtype 来查看，dtype支持：'bool'，'float16'，'float32'，'float64'，'uint8'，'int8'，'int16'，'int32'，'int64'。

* 通过Python元素创建的Tensor，可以通过dtype来进行指定，如果未指定：

    * 对于python整型数据，则会创建int64型Tensor
    * 对于python浮点型数据，默认会创建float32型Tensor，并且可以通过set_default_type来调整浮点型数据的默认类型。

* 通过Numpy array创建的Tensor，则与其原来的dtype保持相同。

```python
print("Tensor dtype from Python integers:", paddle.to_tensor(1).dtype)
print("Tensor dtype from Python floating point:", paddle.to_tensor(1.0).dtype)
```
```text
Tensor dtype from Python integers: VarType.INT64
Tensor dtype from Python floating point: VarType.FP32
```

Paddle提供了**cast**接口来改变dtype：
```python
float32_tensor = paddle.to_tensor(1.0)

float64_tensor = paddle.cast(float32_tensor, dtype='float64')
print("Tensor after cast to float64:", float64_tensor.dtype)

int64_tensor = paddle.cast(float32_tensor, dtype='int64')
print("Tensor after cast to int64:", int64_tensor.dtype)
```
```text
Tensor after cast to float64: VarType.FP64
Tensor after cast to int64: VarType.INT64
```

### Tensor的place

初始化**Tensor**时可以通过**place**来指定其分配的设备位置，可支持的设备位置有三种：CPU/GPU/固定内存，其中固定内存也称为不可分页内存或锁页内存，其与GPU之间具有更高的读写效率，并且支持异步传输，这对网络整体性能会有进一步提升，但其缺点是分配空间过多时可能会降低主机系统的性能，因为其减少了用于存储虚拟内存数据的可分页内存。

* **创建CPU上的Tensor**：
```python
cpu_tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
print(cpu_tensor)
```
```text
Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=True,
       [1])
```

* **创建GPU上的Tensor**：
```python
gpu_tensor = paddle.to_tensor(1, place=paddle.CUDAPlace(0))
print(gpu_tensor)
```
```text
Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [1])
```

* **创建固定内存上的Tensor**：
```python
pin_memory_tensor = paddle.to_tensor(1, place=paddle.CUDAPinnedPlace())
print(pin_memory_tensor)
```

```text
Tensor(shape=[1], dtype=int64, place=CUDAPinnedPlace, stop_gradient=True,
       [1])
```

### Tensor的name

Tensor的name是其唯一的标识符，为python 字符串类型，查看一个Tensor的name可以通过Tensor.name属性。默认地，在每个Tensor创建时，Paddle会自定义一个独一无二的name。

```python
print("Tensor name:", paddle.to_tensor(1).name)
```
```text
Tensor name: generated_tensor_0
```

## Tensor的操作

### 索引和切片
您可以通过索引或切片方便地访问或修改 Tensor。Paddle 使用标准的 Python 索引规则与 Numpy 索引规则，与 [Indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings)类似。具有以下特点：

1. 基于 0-n 的下标进行索引，如果下标为负数，则从尾部开始计算
2. 通过冒号 ``:`` 分隔切片参数 ``start:stop:step`` 来进行切片操作，其中 start、stop、step 均可缺省

#### 访问 Tensor
* 针对1-D **Tensor**，则仅有单个轴上的索引或切片：
```python
ndim_1_tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", ndim_1_tensor.numpy())

print("First element:", ndim_1_tensor[0].numpy())
print("Last element:", ndim_1_tensor[-1].numpy())
print("All element:", ndim_1_tensor[:].numpy())
print("Before 3:", ndim_1_tensor[:3].numpy())
print("From 6 to the end:", ndim_1_tensor[6:].numpy())
print("From 3 to 6:", ndim_1_tensor[3:6].numpy())
print("Interval of 3:", ndim_1_tensor[::3].numpy())
print("Reverse:", ndim_1_tensor[::-1].numpy())
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

* 针对2-D及以上的 **Tensor**，则会有多个轴上的索引或切片：
```python
ndim_2_tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print("Origin Tensor:", ndim_2_tensor.numpy())
print("First row:", ndim_2_tensor[0].numpy())
print("First row:", ndim_2_tensor[0, :].numpy())
print("First column:", ndim_2_tensor[:, 0].numpy())
print("Last column:", ndim_2_tensor[:, -1].numpy())
print("All element:", ndim_2_tensor[:].numpy())
print("First row and second column:", ndim_2_tensor[0, 1].numpy())
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

索引或切片的第一个值对应 axis 0，第二个值对应 axis 1，以此类推，如果某个 axis 上未指定索引，则默认为 ``:`` 。例如：
```python
ndim_2_tensor[1]
ndim_2_tensor[1, :]
```
这两种操作的结果是完全相同的。

```text
Tensor(shape=[4], dtype=int64, place=CPUPlace, stop_gradient=True,
       [4, 5, 6, 7])
```

#### 修改 Tensor

> **注意：**
>
> 请慎重通过索引或切片修改 Tensor，该操作会**原地**修改该 Tensor 的数值，且原值不会被保存。如果被修改的 Tensor 参与梯度计算，将仅会使用修改后的数值，这可能会给梯度计算引入风险。Paddle 之后将会对具有风险的操作进行检测和报错。

与访问 Tensor 类似，修改 Tensor 可以在单个或多个轴上通过索引或切片操作。同时，支持将多种类型的数据赋值给该 Tensor，当前支持的数据类型有：`int`, `float`, `numpy.ndarray`, `Tensor`。

```python
import paddle
import numpy as np

x = paddle.to_tensor(np.ones((2, 3)).astype(np.float32)) # [[1., 1., 1.], [1., 1., 1.]]

x[0] = 0                      # x : [[0., 0., 0.], [1., 1., 1.]]        id(x) = 4433705584
x[0:1] = 2.1                  # x : [[2.1, 2.1, 2.1], [1., 1., 1.]]     id(x) = 4433705584
x[...] = 3                    # x : [[3., 3., 3.], [3., 3., 3.]]        id(x) = 4433705584

x[0:1] = np.array([1,2,3])    # x : [[1., 2., 3.], [3., 3., 3.]]        id(x) = 4433705584

x[1] = paddle.ones([3])       # x : [[1., 2., 3.], [1., 1., 1.]]        id(x) = 4433705584
```

---

同时，Paddle 还提供了丰富的 Tensor 操作的 API，包括数学运算符、逻辑运算符、线性代数相关等100余种 API，这些 API 调用有两种方法：
```python
x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float64")
y = paddle.to_tensor([[5.5, 6.6], [7.7, 8.8]], dtype="float64")

print(paddle.add(x, y), "\n")
print(x.add(y), "\n")
```

```text
Tensor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
       [[6.60000000, 8.80000000],
        [        11., 13.20000000]])

Tensor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
       [[6.60000000, 8.80000000],
        [        11., 13.20000000]])
```

可以看出，使用 **Tensor 类成员函数** 和 **Paddle API** 具有相同的效果，由于 **类成员函数** 操作更为方便，以下均从 **Tensor 类成员函数** 的角度，对常用 **Tensor** 操作进行介绍。

### 数学运算符
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

Paddle对python数学运算相关的魔法函数进行了重写，以下操作与上述结果相同。
```text
x + y  -> x.add(y)            #逐元素相加
x - y  -> x.subtract(y)       #逐元素相减
x * y  -> x.multiply(y)       #逐元素相乘
x / y  -> x.divide(y)         #逐元素相除
x % y  -> x.mod(y)            #逐元素相除并取余
x ** y -> x.pow(y)            #逐元素幂运算
```

### 逻辑运算符
```python
x.isfinite()                  #判断tensor中元素是否是有限的数字，即不包括inf与nan
x.equal_all(y)                #判断两个tensor的全部元素是否相等，并返回shape为[1]的bool Tensor
x.equal(y)                    #判断两个tensor的每个元素是否相等，并返回shape相同的bool Tensor
x.not_equal(y)                #判断两个tensor的每个元素是否不相等
x.less_than(y)                #判断tensor x的元素是否小于tensor y的对应元素
x.less_equal(y)               #判断tensor x的元素是否小于或等于tensor y的对应元素
x.greater_than(y)             #判断tensor x的元素是否大于tensor y的对应元素
x.greater_equal(y)            #判断tensor x的元素是否大于或等于tensor y的对应元素
x.allclose(y)                 #判断tensor x的全部元素是否与tensor y的全部元素接近，并返回shape为[1]的bool Tensor
```

同样地，Paddle对python逻辑比较相关的魔法函数进行了重写，以下操作与上述结果相同。
```text
x == y  -> x.equal(y)         #判断两个tensor的每个元素是否相等
x != y  -> x.not_equal(y)     #判断两个tensor的每个元素是否不相等
x < y   -> x.less_than(y)     #判断tensor x的元素是否小于tensor y的对应元素
x <= y  -> x.less_equal(y)    #判断tensor x的元素是否小于或等于tensor y的对应元素
x > y   -> x.greater_than(y)  #判断tensor x的元素是否大于tensor y的对应元素
x >= y  -> x.greater_equal(y) #判断tensor x的元素是否大于或等于tensor y的对应元素
```

以下操作仅针对bool型Tensor：
```python
x.logical_and(y)              #对两个bool型tensor逐元素进行逻辑与操作
x.logical_or(y)               #对两个bool型tensor逐元素进行逻辑或操作
x.logical_xor(y)              #对两个bool型tensor逐元素进行逻辑亦或操作
x.logical_not(y)              #对两个bool型tensor逐元素进行逻辑非操作
```

### 线性代数相关
```python
x.cholesky()                  #矩阵的cholesky分解
x.t()                         #矩阵转置
x.transpose([1, 0])           #交换axis 0 与axis 1的顺序
x.norm('fro')                 #矩阵的Frobenius 范数
x.dist(y, p=2)                #矩阵（x-y）的2范数
x.matmul(y)                   #矩阵乘法
```
需要注意，Paddle中Tensor的操作符均为非inplace操作，即 ``x.add(y)`` 不会在**tensor x**上直接进行操作，而会返回一个新的**Tensor**来表示运算结果。

更多Tensor操作相关的API，请参考 [class paddle.Tensor](../../../api/paddle/tensor/creation/Tensor_cn.html)
