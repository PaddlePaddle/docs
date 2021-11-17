# 支持语法

## 一、主要针对场景

本文档概览性介绍了飞桨动转静功能的语法支持情况，旨在向用户提供一个便捷的语法速查表，**主要适用于如下场景**：

1. 不确定当前动态图模型是否可以正确转化为静态图

2. 转化过程中出现了问题但不知道如何排查

3. 当出现不支持的语法时，如何修改源码适配动转静语法


若您初次接触动转静功能，或对此功能尚不熟悉，推荐您阅读：[基础接口用法](./basic_usage_cn.html)；

若您想进行预测模型导出，或想深入了解此模块，推荐您阅读：[预测模型导出](./export_model_cn.html)；

若您动静转换遇到了问题，或想学习调试的技巧，推荐您阅读：[报错调试经验](./debugging_cn.html)。

## 二、语法支持速查列表

|分类 |python语法 | 是否<br>支持 | 概要 |
|:---:|:---:|:---:|:---:|
|<font color='purple'>控制流</font>| [if-else](#1) | 支持 | 自适应识别和转为静态图cond接口，或保持python if 执行 |
|| [while](#2) | 支持 |自适应识别和转为静态图while\_loop接口，或保持python while 执行 |
|| [for](#3) | 支持 | `for _ in x`语法支持对Tensor的迭代访问 |
|| [break<br>continue](#4)| 支持 | 支持循环中任意位置的break和continue |
|| [return](#4)| 支持 | 支持循环体中提前return |
|<font color='purple'>运算符</font>| +，-，，/，\*, >, <, >= , <=, == | 支持 | 自适应识别和应用paddle的运算符重载 |
|| [and, or, not](#5) | 支持 | 1.如果运算符两个都是Tensor，会组网静态图。 <br> 2. 如果运算符都不是Tensor，那么使用原始python语义 <br> 3. 如果一个是Tensor一个是非Tensor，那么也会使用python语义，但是结果不会出错。 |
|| [类型转换运算符](#6) | 支持 | 自适应转换为paddle.cast 操作|
|<font color='purple'>Paddle shape</font>| [Tensor.shape()](#9) | 部分支持 | 支持获取编译期shape信息，可能包含-1 |
|<font color='purple'>python函数类</font>| [print(x)](#7) | 支持 | 自适应识别和转为静态图的PrintOp |
|| [len(x)](#) | 支持 | 支持返回Tensor编译期shape[0]的值 |
|| [lambda 表达式](#7) | 支持 | 等价转换 |
|| [函数调用其他函数](#7) | 支持 | 会对内部的函数递归地进行动转静 |
|| [函数递归调用](#7) | 不支持 | 递归调用不会终止 |
|| [list sort](#8) | 不支持 | list可能会被转化为TensorArray，故不支持此复杂操作 |
|<font color='purple'>报错异常相关</font>| assert | 支持 | 自适应识别和转换为静态图Assert接口 | 无 |
|<font color='purple'>Python基本容器</font>| [list](#8) | 部分支持 | 在控制流中转化为TensorArray，支持append，pop |
|| [Dict](#8) | 支持 | 原生支持 |
|<font color='purple'>第三方库相关</font>| numpy | 部分支持 | 仅支持numpy操作不需要导出到Program| 无 |






## 三、详细说明


### 3.1 if-else
<span id='1'></span>

**主要逻辑：**
在动态图中，模型代码是一行一行解释执行的，因此控制流的条件变量是在运行期确定的，意味着False的逻辑分支不会被执行。

在静态图中，控制流通过`cond`接口实现。每个分支分别通过`true_fn` 和`false_fn` 来表示。

当 `if`中的`条件`是`Tensor`时，动转静会自动把该`if-elif-else`语句转化为静态图的`cond` API语句。

当`if`中的`条件`不是`Tensor`时，会按普通Python if-else的逻辑运行。

>注：当`条件`为`Tensor`时，只接受`numel()==1`的bool Tensor，否则会报错。

**错误修改指南：**

 当模型代码中的`if-else`转换或执行报错时，可以参考如下方式排查：

- 使用`if`语句时，请确定`条件变量`是否是`Paddle.Tensor`类型。若不是Tensor类型，则**会当按照常规的python逻辑执行，而不会转化为静态图。**

- 若`if`中`条件变量`为Tensor类型，需确保其为boolean类型，且 `tensor.numel()`为1。



### 3.2 while循环
<span id='2'></span>

**主要逻辑：**

当 `while` 循环中的条件是Tensor时，动转静会把该while语句转化为静态图中的`while_loop` API语句，否则会按普通Python while运行。

> 注：while循环条件中的Tensor须是numel为1的bool Tensor，否则会报错。

**错误修改指南：**

类似` if-elif-else`，注意事项也相似。



### 3.3 for 循环
<span id='3'></span>

**主要逻辑：**

for循环按照使用方法的不同，语义有所不同。正常而言，for循环的使用分为如下种类：

- `for _ in range(len) `循环：动转静会先将其转化为等价的Python while循环，然后按while循环的逻辑进行动静转换。

- `for _ in x `循环： 当x是Python容器或迭代器，则会用普通Python逻辑运行。当x是Tensor时，会转化为依次获取x[0], x[1], ... 。
- `for idx, val in enumerate(x)`循环：当x是Python容器或迭代器，则会用普通Python逻辑运行。当x是Tensor时，idx会转化为依次0，1，...的1-D Tensor。val会转化为循环中每次对应拿出x[0], x[1], ... 。

从实现而言，for循环最终会转化为对应的while语句，然后使用`WhileOp`来进行组网。

**使用样例**：

此处使用上述For的第二个用法举例。如果x是一个多维Tensor，则也是返回 x[0] ，x[1]. ...


```python
def ForTensor(x):
    """Fetch element in x and print the square of each x element"""
    for i in x :
        print (i * i)

#调用方法，ForTensor(paddle.to_tensor(x))
```



### 3.4 流程控制语句说明 (return / break / continue)

<span id='4'></span>

**主要逻辑：**

目前的动转静支持for、while等循环中添加break，continue语句改变控制流，也支持在循环内部任意位置添加return语句，支持return不同长度tuple和不同类型的Tensor。

**使用样例**：
```python
# break 的使用样例
def break_usage(x):
    tensor_idx = -1  
    for idx, val in enumerate(x) :
        if val == 2.0 :
            tensor_idx = idx
            break  # <------- jump out of while loop when break ;
    return tensor_idx
```
当时输入 x = Tensor([1.0, 2.0 ,3.0]) 时，输出的tensor_idx是 Tensor([1])。

> 注：这里虽然idx是-1，但是返回值还是Tensor。因为`tensor_idx` 在 while loop中转化为了`Tensor`。


### 3.5 与、或、非
<span id='5'></span>

**主要逻辑：**

动转静模块支持将与、或、非三种运算符进行转换并动态判断，按照两个运算符x和y的不同，会有不同的语义：

- 如果运算符两个都是Tensor，会组网静态图。

- 如果运算符都不是Tensor，那么使用原始python语义。

- 如果一个是Tensor，那么会走默认的python语义（最后还是tensor的运算符重载结果）

> 注：若按照paddle的语义执行，与、或、非不再支持lazy模式，意味着两个表达式都会被eval，而不是按照x的值来判断是否对y进行eval。

**使用样例**：

```python
def and(x, y):
    z = y and x
    return z
```

### 3.6 类型转换运算符
<span id='6'></span>
**主要逻辑：**

动态图中可以直接用Python的类型转化语法来转化Tensor类型。如若x是Tensor时，float(x)可以将x的类型转化为float。

动转静在运行时判断x是否是Tensor，若是，则在动转静时使用静态图`cast`接口转化相应的Tensor类型。

**使用样例**：

```python
def float_convert(x):
    z = float(x)
    return z
# 如果输入是 x = Tensor([True]) ，则 z = Tensor([1.0])
```


### 3.7 对一些python函数调用的转换
<span id='7'></span>
**主要逻辑：**

动转静支持大部分的python函数调用。函数调用都会被统一包装成为`convert_xxx()`的形式，在函数运行期判别类型。若是Paddle类型，则转化为静态图的组网；反之则按照原来的python语义执行。常见函数如下：

- print函数
若参数是Tensor，在动态图模式中print(x)可以打印x的值。动转静时会转化为静态图的Print接口实现；若参数不是Tensor，则按照Python的print语句执行。

- len 函数
若x是Tensor，在动态图模式中len(x)可以获得x第0维度的长度。动转静时会转化为静态图shape接口，并返回shape的第0维。若x是个TensorArray，那么len(x)将会使用静态图接口`control_flow.array_length`返回TensorArray的长度；对于其他情况，会按照普通Python len函数运行。

- lambda 表达式
动转静允许写带有Python lambda表达式的语句，并且我们会适当改写使得返回对应结果。

- 函数内再调用函数（非递归调用）
对于函数内调用其他函数的情况，动转静会对内部的函数递归地进行识别和转写，以实现在最外层函数只需加一次装饰器即可的效果。

**使用样例**：

这里以lambda函数为例，展示使用方法

```python
def lambda_call(x):
    t = lambda x : x * x
    z = t(x)
    return z
# 如果输入是 x = Tensor([2.0]) ，则 z = Tensor([4.0])
```

**不支持用法**：

- 函数的递归调用

动转静暂不支持一个函数递归调用本身。原因是递归常常会用 if-else 构造停止递归的条件。此停止条件在静态图下只是一个 cond 组网，并不能在编译阶段得到递归条件的具体值，会导致函数运行时一直组网递归直至栈溢出。

```python
def recur_call(x):
    if x > 10:
        return x
    return recur_call(x * x) # < ------ 如果输入是 x = Tensor([2.0]) ，动态图输出为 Tensor([16])，静态图会出现调用栈溢出
```

### 3.8 List和Dict容器
<span id='8'></span>
**主要逻辑：**

- List : 若一个list的元素都是Tensor，动转静将其转化为TensorArray。静态图TensorArray仅支持append，pop，修改操作，其他list操作（如sort）暂不支持。若并非所有元素是Tensor，动转静会将其作为普通Python list运行。

- Dict : 动转静支持原生的Python dict 语法。

> 注：List 不支持多重嵌套和其他的操作。具体错误案例见下面**不支持用法**。

**使用样例**：
```python
def list_example(x, y):
     a = [ x ]   # < ------ 支持直接创建
     a.append(x) # < ------ 支持调用append、pop操作
     a[1] = y    # < ------ 支持下标修改append
     return a[0] # < ------ 支持下标获取
```

**不支持用法**：

- List的多重嵌套

 如 `l = [[tensor1, tensor2], [tensor3, tensor4]] `，因为现在动转静将元素全是Tensor的list转化为TensorArray，但TensorArray还不支持多维数组，因此这种情况下，动转静无法正确运行。遇到这类情况我们建议尽量用一维list，或者自己使用PaddlePaddle的create_array，array_read，array_write接口编写为TensorArray。


- List的其他的操作，例如sort之类

```python
# 不支持的 list sort 操作
def sort_list(x, y):
    a = [x, y]
    sort(a)   # < -----  不支持，因为转化为TensorArray之后不支持sort操作。但是支持简单的append,pop和按下标修改
    return a
```

### 3.9 paddle shape函数
<span id='9'></span>
**主要逻辑：**

动转静部分支持shape函数：

- 【支持】当直接简单的使用shape时，可以正确获取tensor的shape。



- 【不支持】当直接使用支持改变变量的shape后(例如reshape操作)调用其shape作为PaddlePaddle API参数。

 如 `x = reshape(x, shape=shape_tensor) `，再使用 x.shape[0] 的值进行其他操作。这种情况会由于动态图和静态图的本质不同而使得动态图能够运行，但静态图运行失败。其原因是动态图情况下，API是直接返回运行结果，因此 x.shape 在经过reshape运算后是确定的。但是在转化为静态图后，因为静态图API只是组网，shape_tensor 的值在组网时是不知道的，所以 reshape 接口组网完，静态图并不知道 x.shape 的值。PaddlePaddle静态图用-1表示未知的shape值，此时 x 的shape每个维度会被设为-1，而不是期望的值。同理，类似expand等更改shape的API，其输出Tensor再调用shape也难以进行动转静。

**使用样例**：

```python
def get_shape(x):
    return x.shape[0]
```

**不支持用法举例**：
```
def error_shape(x, y):
    y = y.cast('int32')
    t = x.reshape(y)
    return t.shape[0] # <------- 输入在x = Tensor([2.0, 1.0])，y = Tensor([2])时，动态图输出为2，而静态图输出为 -1 。不支持
```
