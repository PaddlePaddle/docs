# 支持语法

## 一、主要针对场景

本文档概览性介绍了飞桨动转静功能的语法支持情况，旨在提供一个便捷的语法速查表，**主要适用于如下场景**：

1. 不确定当前动态图模型是否可以正确转化为静态图

2. 转化过程中出现了问题但不知道如何排查

3. 当出现不支持的语法时，如何修改源码适配动转静语法


若初次接触动转静功能，或对此功能尚不熟悉，推荐阅读：[使用样例](./basic_usage_cn.html)；

若动静转换遇到了问题，或想学习调试的技巧，推荐阅读：[报错调试](./debugging_cn.html)。

## 二、语法支持速查列表

|分类 |python 语法 | 是否<br>支持 | 概要 |
|:---:|:---:|:---:|:---:|
|<font color='purple'>控制流</font>| [if-else](#1) | 支持 | 自适应识别和转为静态图 cond 接口，或保持 python if 执行 |
|| [while](#2) | 支持 |自适应识别和转为静态图 while\_loop 接口，或保持 python while 执行 |
|| [for](#3) | 支持 | `for _ in x`语法支持对 Tensor 的迭代访问 |
|| [break<br>continue](#4)| 支持 | 支持循环中任意位置的 break 和 continue |
|| [return](#4)| 支持 | 支持循环体中提前 return |
|<font color='purple'>运算符</font>| +，-，，/，\*, >, <, >= , <=, == | 支持 | 自适应识别和应用 paddle 的运算符重载 |
|| [and, or, not](#5) | 支持 | 1.如果运算符两个都是 Tensor，会组网静态图。 <br> 2. 如果运算符都不是 Tensor，那么使用原始 python 语义 <br> 3. 如果一个是 Tensor 一个是非 Tensor，那么也会使用 python 语义，但是结果不会出错。 |
|| [类型转换运算符](#6) | 支持 | 自适应转换为 paddle.cast 操作|
|<font color='purple'>Paddle shape</font>| [Tensor.shape()](#9) | 部分支持 | 支持获取编译期 shape 信息，可能包含-1 |
|<font color='purple'>python 函数类</font>| [print(x)](#7) | 支持 | 自适应识别和转为静态图的 PrintOp |
|| [len(x)](#) | 支持 | 支持返回 Tensor 编译期 shape[0]的值 |
|| [lambda 表达式](#7) | 支持 | 等价转换 |
|| [函数调用其他函数](#7) | 支持 | 会对内部的函数递归地进行动转静 |
|| [函数递归调用](#7) | 不支持 | 递归调用不会终止 |
|| [list sort](#8) | 不支持 | list 可能会被转化为 TensorArray，故不支持此复杂操作 |
|<font color='purple'>报错异常相关</font>| assert | 支持 | 自适应识别和转换为静态图 Assert 接口 | 无 |
|<font color='purple'>Python 基本容器</font>| [list](#8) | 部分支持 | 在控制流中转化为 TensorArray，支持 append，pop |
|| [Dict](#8) | 支持 | 原生支持 |
|<font color='purple'>第三方库相关</font>| numpy | 部分支持 | 仅支持 numpy 操作不需要导出到 Program| 无 |






## 三、详细说明


### 3.1 if-else
<span id='1'></span>

**主要逻辑：**
在动态图中，模型代码是一行一行解释执行的，因此控制流的条件变量是在运行期确定的，意味着 False 的逻辑分支不会被执行。

在静态图中，控制流通过`cond`接口实现。每个分支分别通过`true_fn` 和`false_fn` 来表示。

当 `if`中的`条件`是`Tensor`时，动转静会自动把该`if-elif-else`语句转化为静态图的`cond` API 语句。

当`if`中的`条件`不是`Tensor`时，会按普通 Python if-else 的逻辑运行。

>注：当`条件`为`Tensor`时，只接受`numel()==1`的 bool Tensor，否则会报错。

**错误修改指南：**

 当模型代码中的`if-else`转换或执行报错时，可以参考如下方式排查：

- 使用`if`语句时，请确定`条件变量`是否是`Paddle.Tensor`类型。若不是 Tensor 类型，则**会当按照常规的 python 逻辑执行，而不会转化为静态图。**

- 若`if`中`条件变量`为 Tensor 类型，需确保其为 boolean 类型，且 `tensor.numel()`为 1。



### 3.2 while 循环
<span id='2'></span>

**主要逻辑：**

当 `while` 循环中的条件是 Tensor 时，动转静会把该 while 语句转化为静态图中的`while_loop` API 语句，否则会按普通 Python while 运行。

> 注：while 循环条件中的 Tensor 须是 numel 为 1 的 bool Tensor，否则会报错。

**错误修改指南：**

类似` if-elif-else`，注意事项也相似。



### 3.3 for 循环
<span id='3'></span>

**主要逻辑：**

for 循环按照使用方法的不同，语义有所不同。正常而言，for 循环的使用分为如下种类：

- `for _ in range(len) `循环：动转静会先将其转化为等价的 Python while 循环，然后按 while 循环的逻辑进行动静转换。

- `for _ in x `循环： 当 x 是 Python 容器或迭代器，则会用普通 Python 逻辑运行。当 x 是 Tensor 时，会转化为依次获取 x[0], x[1], ... 。
- `for idx, val in enumerate(x)`循环：当 x 是 Python 容器或迭代器，则会用普通 Python 逻辑运行。当 x 是 Tensor 时，idx 会转化为依次 0，1，...的 1-D Tensor。val 会转化为循环中每次对应拿出 x[0], x[1], ... 。

从实现而言，for 循环最终会转化为对应的 while 语句，然后使用`WhileOp`来进行组网。

**使用样例**：

此处使用上述 For 的第二个用法举例。如果 x 是一个多维 Tensor，则也是返回 x[0] ，x[1]. ...


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

目前的动转静支持 for、while 等循环中添加 break，continue 语句改变控制流，也支持在循环内部任意位置添加 return 语句，支持 return 不同长度 tuple 和不同类型的 Tensor。

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
当时输入 x = Tensor([1.0, 2.0 ,3.0]) 时，输出的 tensor_idx 是 Tensor([1])。

> 注：这里虽然 idx 是-1，但是返回值还是 Tensor。因为`tensor_idx` 在 while loop 中转化为了`Tensor`。


### 3.5 与、或、非
<span id='5'></span>

**主要逻辑：**

动转静模块支持将与、或、非三种运算符进行转换并动态判断，按照两个运算符 x 和 y 的不同，会有不同的语义：

- 如果运算符两个都是 Tensor，会组网静态图。

- 如果运算符都不是 Tensor，那么使用原始 python 语义。

- 如果一个是 Tensor，那么会走默认的 python 语义（最后还是 tensor 的运算符重载结果）。

> 注：若按照 paddle 的语义执行，与、或、非不再支持 lazy 模式，意味着两个表达式都会被 eval，而不是按照 x 的值来判断是否对 y 进行 eval。

**使用样例**：

```python
def and(x, y):
    z = y and x
    return z
```

### 3.6 类型转换运算符
<span id='6'></span>
**主要逻辑：**

动态图中可以直接用 Python 的类型转化语法来转化 Tensor 类型。如若 x 是 Tensor 时，float(x)可以将 x 的类型转化为 float。

动转静在运行时判断 x 是否是 Tensor，若是，则在动转静时使用静态图`cast`接口转化相应的 Tensor 类型。

**使用样例**：

```python
def float_convert(x):
    z = float(x)
    return z
# 如果输入是 x = Tensor([True]) ，则 z = Tensor([1.0])
```


### 3.7 对一些 python 函数调用的转换
<span id='7'></span>
**主要逻辑：**

动转静支持大部分的 python 函数调用。函数调用都会被统一包装成为`convert_xxx()`的形式，在函数运行期判别类型。若是 Paddle 类型，则转化为静态图的组网；反之则按照原来的 python 语义执行。常见函数如下：

- print 函数
若参数是 Tensor，在动态图模式中 print(x)可以打印 x 的值。动转静时会转化为静态图的 Print 接口实现；若参数不是 Tensor，则按照 Python 的 print 语句执行。

- len 函数
若 x 是 Tensor，在动态图模式中 len(x)可以获得 x 第 0 维度的长度。动转静时会转化为静态图 shape 接口，并返回 shape 的第 0 维。若 x 是个 TensorArray，那么 len(x)将会使用静态图接口`control_flow.array_length`返回 TensorArray 的长度；对于其他情况，会按照普通 Python len 函数运行。

- lambda 表达式
动转静允许写带有 Python lambda 表达式的语句，并且我们会适当改写使得返回对应结果。

- 函数内再调用函数（非递归调用）
对于函数内调用其他函数的情况，动转静会对内部的函数递归地进行识别和转写，以实现在最外层函数只需加一次装饰器即可的效果。

**使用样例**：

这里以 lambda 函数为例，展示使用方法

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

### 3.8 List 和 Dict 容器
<span id='8'></span>
**主要逻辑：**

- List : 若一个 list 的元素都是 Tensor，动转静将其转化为 TensorArray。静态图 TensorArray 仅支持 append，pop，修改操作，其他 list 操作（如 sort）暂不支持。若并非所有元素是 Tensor，动转静会将其作为普通 Python list 运行。

- Dict : 动转静支持原生的 Python dict 语法。

> 注：List 不支持多重嵌套和其他的操作。具体错误案例见下面**不支持用法**。

**使用样例**：
```python
def list_example(x, y):
     a = [ x ]   # < ------ 支持直接创建
     a.append(x) # < ------ 支持调用 append、pop 操作
     a[1] = y    # < ------ 支持下标修改 append
     return a[0] # < ------ 支持下标获取
```

**不支持用法**：

- List 的多重嵌套

 如 `l = [[tensor1, tensor2], [tensor3, tensor4]] `，因为现在动转静将元素全是 Tensor 的 list 转化为 TensorArray，但 TensorArray 还不支持多维数组，因此这种情况下，动转静无法正确运行。遇到这类情况我们建议尽量用一维 list，或者自己使用 PaddlePaddle 的 create_array，array_read，array_write 接口编写为 TensorArray。


- List 的其他的操作，例如 sort 之类

```python
# 不支持的 list sort 操作
def sort_list(x, y):
    a = [x, y]
    sort(a)   # < -----  不支持，因为转化为 TensorArray 之后不支持 sort 操作。但是支持简单的 append,pop 和按下标修改
    return a
```

### 3.9 paddle shape 函数
<span id='9'></span>
**主要逻辑：**

动转静部分支持 shape 函数：

- 【支持】当直接简单的使用 shape 时，可以正确获取 tensor 的 shape。



- 【不支持】当直接使用支持改变变量的 shape 后(例如 reshape 操作)调用其 shape 作为 PaddlePaddle API 参数。

 如 `x = reshape(x, shape=shape_tensor) `，再使用 x.shape[0] 的值进行其他操作。这种情况会由于动态图和静态图的本质不同而使得动态图能够运行，但静态图运行失败。其原因是动态图情况下，API 是直接返回运行结果，因此 x.shape 在经过 reshape 运算后是确定的。但是在转化为静态图后，因为静态图 API 只是组网，shape_tensor 的值在组网时是不知道的，所以 reshape 接口在组网完成后，静态图并不知道 x.shape 的值。PaddlePaddle 静态图用-1 表示未知的 shape 值，此时 x 的 shape 每个维度会被设为-1，而不是期望的值。同理，类似 expand 等更改 shape 的 API，其输出 Tensor 再调用 shape 也难以进行动转静。

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
    return t.shape[0] # <------- 输入在 x = Tensor([2.0, 1.0])，y = Tensor([2])时，动态图输出为 2，而静态图输出为 -1 。不支持
```
