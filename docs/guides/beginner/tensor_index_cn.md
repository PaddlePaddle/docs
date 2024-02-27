# Paddle Tensor 索引介绍


## 1. Tensor 索引简介
索引是 Paddle 中针对 Tensor 一个非常重要的操作，它能够实现 Tensor 子元素的访问取值（`__getitem__`方法）和修改（`__setitem__`方法）。飞桨框架支持标准的 Python 索引规则及进一步的扩展，在模型组网中，相比于通过多个一般 API 组合实现，操作能够更简单及直观。

对于一个 Tensor `x`，`index`指明了想要访问的位置。根据`index`的类型不同，可以分为以下类型的索引场景：
| 场景 | 基础索引 | 高级索引 | 联合索引 |
| :--- | :---: | :---: | :---: |
| 取值(`__getitem__`) | · **y = x[0, 2:4]** <br>等价于: <br>y = paddle.slice(x, [0,1], [0,2], [1,4], decrease_axes=[1]) | · **y = x[[0,1], [2,3]]** <br>等价于：<br>index = paddle.stack([Tensor([0,1]), Tensor([2,3]), axis=1) y = paddle.gather_nd(x, index) | · **y = x[0, [0,2], ..., 2:5:2, None]** <br>等价替换超过 10 行代码 |
| 赋值(`__setitem__`) | · **x[0, 2:3] = Tensor(1.0)** <br>等价于：<br>paddle.slice_scatter_(x, [0,1], [0,2], [1,4], decrease_axes=[1]) | · **x[[0,1], [2,3]] = Tensor(1.0)** <br>等价于：<br>paddle.index_put_(x, ([Tensor([0,1]), Tensor([2,3]), Tensor(1.0)) | ·  **x[0, [0,2], ..., 2:5:2, None] = 1.0** <br>等价替换超过 10 行代码|

**注意：** 与 Numpy 等其他主流框架一致，在 Paddle 中用元组(tuple)来表示打包后的索引对象`index`，元组内部的每个元素分别表示对应轴的索引内容，即：
```python
x[(index_1, index_2, ..., index_n)] == x[index_1, index_2, ..., index_n]
```

## 2. 基础索引(Basic Index)
### 2.1 简介
当`index`中的所有元素均属于下列类型时，称为基础索引：
- 整形或表示整数的 0-D `Tensor/Ndarray`
- Python `slice`对象，即`:`或`::`
- Python `Ellipsis`对象，即`...`
- Python `None`类型

**注意：**
在**动态图模式**下，通过基础索引取值时，输出将是**原 Tensor 的 view**，即如果对输出 Tensor 再进行修改，会影响原 Tensor 的值。而在**静态图模式**下，输出是一个新的 Tensor。由于在两种模式下存在差异，**请谨慎使用这个特性**。
```python
# In Paddle dynamic mode
>>> a = paddle.ones((2,3))
>>> a
Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.]])
>>> b = a[0]   # b is a view of a
>>> b
Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [1., 1., 1.])
>>> b[1] = 10  # modifacation of b will affect a
>>> b
Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [1. , 10., 1. ])
>>> a
Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[1. , 10., 1. ],
        [1. , 1. , 1. ]])
```

### 2.2 整形或表示整数的 0-D Tensor/Ndarray
这个场景与 Python 原生类型的索引规则类似，表示选择对应轴上的具体位置的元素，从 0 开始计数，也可以接收负数作为输入，表示从对应轴的最后开始计数。在取值场景中，由于指定轴仅选择了单个元素，因此该轴对应的维度将被消减。
```python
>>> a = paddle.arange(6).reshape((2,3))
>>> a
Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1, 2],
        [3, 4, 5]])
>>> b = a[1]  # select the second row in first axis
>>> b
Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [3, 4, 5])
>>> c = a[-1] # select the last row in first axis
>>> c
Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [3, 4, 5])
```
特别的，如果所有轴上同时选取单个元素，则最终结果中所有轴的维度都将被消减，返回一个 0-D Tensor，而非一个 Scalar 类型。
```python
>>> d = a[1, 0]
>>> d
Tensor(shape=[], dtype=int64, place=Place(cpu), stop_gradient=True,
       3)
```

从 Paddle 2.5 版本开始，使用 0-D Tensor 而非 1-D Tensor 表示 Scalar 语义。在索引时，0-D Tensor 与其对应的数值语义是一致的。
```python
>>> b = a[1]
>>> b
Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [3, 4, 5])

>>> index = paddle.full([], 1, dtype='int32')
>>> c = a[index]
>>> c
Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [3, 4, 5])
```

### 2.3 Python `slice`对象
`slice`对象由 start/end/step 定义，这个场景与 Python 原生类型的索引规则类似，表示在对应轴上的起始-结束区间`[start, end)`内，根据指定的步长`step`进行切片选取。对于 start/end/step 同样可以是对应的 0-D Tensor/Ndarray，也可以是负数。当为负数时，start/end 表示从对应轴的最后开始计数，step 为负数时，表示逆序选取。在取值场景中，该轴对应的维度将被保留，大小为选取到的元素数量。

```python
>>> a = paddle.arange(8).reshape((4,2))
>>> a
Tensor(shape=[4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])
>>> b = a[0:2]  # select elements [0, 2) with step 1 in first axis
>>> b
Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [2, 3]])
>>> c = a[::2]  # select elements [0, 4) with step 2 in first axis
>>> c
Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [4, 5]])
>>> d = a[::-1] # reversed selection in first axis
>>> d
Tensor(shape=[4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[6, 7],
        [4, 5],
        [2, 3],
        [0, 1]])
```

### 2.4 Python `Ellipsis`对象
省略号对象`...`是多个连续的 slice 对象`:`的简写，可以出现在索引中任意位置，但只能出现一次，表示对所表示的单个或多个轴进行全选切片。在实际使用时，会根据省略号前后的索引情况推断出所代表的轴。
```python
>>> a = paddle.arange(8).reshape((2,2,2))
>>> a
Tensor(shape=[2, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[0, 1],
         [2, 3]],

        [[4, 5],
         [6, 7]]])
>>> b = a[...] # ... covers axes 0,1,2, equals a[:,:,:], which means select all elements
>>> b
Tensor(shape=[2, 2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[0, 1],
         [2, 3]],

        [[4, 5],
         [6, 7]]])
>>> c = a[1, ...]  # ... covers axes 1,2, equals a[1,:,:]
>>> c
Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[4, 5],
        [6, 7]])
>>> d = a[1, ..., 0] # ... covers axis 1, equals a[1,:,0]
>>> d
Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [4, 6])
```

### 2.5 Python `None`类型
`None`（或`np.newaxis`类型，两者实质是相同的），通常在取值场景中使用，表示取值的结果在对应位置扩展大小为 1 的维度。
```python
>>> a = paddle.arange(8).reshape((2,4))
>>> a
Tensor(shape=[2, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1, 2, 3],
        [4, 5, 6, 7]])
>>> b = a[:, None]
>>> b
Tensor(shape=[2, 1, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[0, 1, 2, 3]],

        [[4, 5, 6, 7]]])
```

## 3. 高级索引（Advanced Indexing）
### 3.1 简介
当`index`中的所有元素均属于下列类型时，称为高级索引：
- 整数类型的非 0-D `Tensor/Ndarray`或 Python `List`
- bool 类型的`Tensor/Ndarray`或 Python `List`
- Python `bool`
- 至少包含一个上述类型的 Python `Tuple`

高级索引是主流框架在 Python 原生类型的索引上的进一步扩展，支持更多非均匀选择的场景，具有更高的灵活性。根据索引中数据类型的不同，主要分为**整形索引**和**布尔索引**两类。和基础索引不同的是，在取值场景中，高级索引将会**返回一个全新的 Tensor**，修改该 Tensor 不会影响原始 Tensor。
```python
>>> a = paddle.ones((2,3))
>>> a
Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.]])
>>> b = a[[0]] # b is not a view of a
>>> b
Tensor(shape=[1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[1., 1., 1.]])
>>> b[0] = 10  # modify b will not affect a
>>> b
Tensor(shape=[1, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[10., 10., 10.]])
>>> a
Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.]])
```
**注意：**
在飞桨中，Python `List`类型，或是包含序列的 Python `Tuple`类型，其语义与其对应的`Tensor/Ndarray`一致，在框架内部均会先转换为`Tensor`。

### 3.2 整形索引
整形索引允许根据给定的`index`，对 Tensor 中的元素进行任意选择并进行组合。这在某些非均匀选择场景下非常有用（如选择出某些特定的 id 对应的 embedding 向量）。
```python
>>> a = paddle.arange(8).reshape((4,2))
>>> a
Tensor(shape=[4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])
>>> b = a[[0,2,1]]  # select rows 0,2 and 1 in first axis
>>> b
Tensor(shape=[3, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [4, 5],
        [2, 3]])

>>> c = a[[0,1,0]]  # row 0 was selected twice
>>> c
Tensor(shape=[3, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [2, 3],
        [0, 1]])

>>> index = paddle.to_tensor([[1], [2]])
>>> d = a[index]  # select rows 1 and 2 in first axis, and combine them according `index`
>>> d
Tensor(shape=[2, 1, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[2, 3]],

        [[4, 5]]])
```

在不同轴同时出现整形索引时，将会通过**广播机制（Broadcasting)** 处理，再根据指定的索引值进行选择，即满足：
```python
output[i_1, ..., i_m] == x[index_1[i_1, ..., i_m], index_2[i_1, ..., i_m],
                           ..., index_n[i_1, ..., i_m]]
```
其中`index_1, ..., index_n`为各轴上的整形索引，经过广播机制变为同一个形状。`i_1, ..., i_m`为广播后的各轴索引位置。
在下面的例子中，[0,2,1]和[0]均是整形索引，因此首先将广播成相同形状，即[0,2,1]和[0,0,0]，再逐个选择出 a[0,0]、a[2,0] 和 a[1,0]。
```python
>>> e = a[[0,2,1], [0]]
>>> e
Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [0, 4, 2])
```

如果不满足广播机制则会报错。
```python
>>> f = a[[0,2,1], [0,1]]  # shape (3,) and (2,) cannot be broadcast together
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.8/dist-packages/paddle/base/dygraph/tensor_patch_methods.py", line 992, in __getitem__
    return self._getitem_dygraph(item)
ValueError: (InvalidArgument) Broadcast dimension mismatch. Operands could not be broadcast together with the shape of X = [2] and the shape of Y = [3]. Received [2] in X is not equal to [3] in Y at i:0.
  [Hint: Expected x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 || y_dims_array[i] <= 1 == true, but received x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 || y_dims_array[i] <= 1:0 != true:1.] (at /workspace/workspace2/Paddle/paddle/fluid/operators/common_infer_shape_functions.cc:80)
```

### 3.3 布尔索引
布尔索引即选择出符合条件为`True`的所有元素，这类似于掩码(mask)的语义。**注意如果在取值时没有符合条件的元素，那么输出的形状将会含有 0，即 0-Size Tensor（不包含具体的数据）**。根据`index`类型的差异，布尔索引有如下进一步的细分场景。

#### 3.3.1 index 为 bool 的 Tensor/Ndarray/List 等类型
当`index`为`bool`类型的 Tensor/Ndarray/List 等类型时，要求在形状上满足下列条件：
- `index`的 rank 小于或等于被索引的 Tensor 的 rank
- `index`的所有轴均与被索引的 Tensor 在对应维度上大小一致


在这个场景下，布尔索引可以通过 nonzero()方法实现与整形索引的转换，即满足：
```python
# nonzero() returns the index of the non-zero elements on each axis
x[bool_index] == paddle.gather_nd(x, bool_index.nonzero())

>>> a = paddle.arange(8).reshape((4,2))
>>> a
Tensor(shape=[4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])
>>> bool_mask = a > 4  # nonzero results are [2,1], [3,0] and [3,1]
>>> a[bool_mask]
Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [5, 6, 7])

>>> a[[True, False,True,False]]  # select row 0 and 2 in first axis
Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [4, 5]])
```

#### 3.3.2 index 为 Python bool 类型
当`index`是一个单独的 Python `bool`类型时，等价于额外添加一个维度，再根据`index`进行选择，即满足：
```python
x[py_bool_index] == x.unsqueeze(0)[[py_bool_index]]

>>> a = paddle.arange(8).reshape((4,2))
>>> a
Tensor(shape=[4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])
>>> a[True]
Tensor(shape=[1, 4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[0, 1],
         [2, 3],
         [4, 5],
         [6, 7]]])
>>> a[False]   # output is 0-Size Tensor
Tensor(shape=[0, 4, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [])
```
## 4. 联合索引（Combined Indexing）
`index`中元素的类型同时包含基础索引和高级索引的类型的场景，称为联合索引。在取值场景下，联合索引和高级索引一样，将会返回一个新的 Tensor。

### 4.1 联合索引的基本计算逻辑
联合索引将会按照**先基础索引，再高级索引**的顺序进行。
```python
>>> a = paddle.arange(24).reshape((2,3,4))
>>> a
Tensor(shape=[2, 3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[0 , 1 , 2 , 3 ],
         [4 , 5 , 6 , 7 ],
         [8 , 9 , 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
>>> b = a[0,[1,2],2]   # This is same with (1) tmp = a[0,:,2] (2) b = tmp[[1,2]]
>>> b
Tensor(shape=[2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [6 , 10])
```

### 4.2 多个高级索引时的计算逻辑
在`index`中同时存在多个高级索引类型时，同样会通过 3.2 节介绍的广播规则确定最终的输出的大小。此外，在取值场景下，还需要额外考虑这些高级索引类型是否相邻，来确定最后输出所处的维度位置。
#### 场景 1-高级索引位置相邻
所有高级索引位置相邻，则最终的输出结果会放在`index`中第一个高级索引出现的位置上。
```python
>>> a = paddle.arange(24).reshape((1,2,3,4))
>>> a
Tensor(shape=[1, 2, 3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[[0 , 1 , 2 , 3 ],
          [4 , 5 , 6 , 7 ],
          [8 , 9 , 10, 11]],

         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]]]])

>>> b = a[:, [0,0,1], [1,2,0],:] # the new dimention is at axis 1
>>> b
Tensor(shape=[1, 3, 4], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[4 , 5 , 6 , 7 ],
         [8 , 9 , 10, 11],
         [12, 13, 14, 15]]])

>>> c = a[:,[0,0,1], [1,2,0], [2,1,0]]
>>> c
Tensor(shape=[1, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[6 , 9 , 12]])
```
#### 场景 2-高级索引位置不相邻
当高级索引位置不相邻时，则`index`对应产生最终的输出结果会放到第一维上。
```python
>>> d = a[:, [1], :, [2,1,0]] # advanced indexes are not adjacent, the new dimention is at axis 0
>>> d
Tensor(shape=[3, 1, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[[14, 18, 22]],

        [[13, 17, 21]],

        [[12, 16, 20]]])
```
## 5. 针对赋值的额外说明
### 5.1 赋值操作的规则
赋值和取值遵循相同的索引规则，因此前面所介绍的各项索引规则均同样生效于赋值场景，区别是赋值除待索引 Tensor `x`和索引`index`外，还包含了值`value`。在 Paddle 中，`value`支持下列类型的输入：
- Python Scalar (如 float / int / complex 等)
- 0-D Tensor/Ndarray，表示 Scalar 语义
- 非 0-D 的 Tensor/Ndarray，要求`value`的形状可广播到`x[index]`取值结果的形状

```python
>>> a = paddle.ones((2,3,4))
>>> a[:,:,2] = 10  # value is Python Scalar
>>> a
Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[[1. , 1. , 10., 1. ],
         [1. , 1. , 10., 1. ],
         [1. , 1. , 10., 1. ]],

        [[1. , 1. , 10., 1. ],
         [1. , 1. , 10., 1. ],
         [1. , 1. , 10., 1. ]]])

>>> a[:,:,1] = paddle.full([], 2) # value is 0-D Scalar Tensor
>>> a
Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[[1. , 2. , 10., 1. ],
         [1. , 2. , 10., 1. ],
         [1. , 2. , 10., 1. ]],

        [[1. , 2. , 10., 1. ],
         [1. , 2. , 10., 1. ],
         [1. , 2. , 10., 1. ]]])

>>> a[:,:,3] = paddle.full([2,1], 5) # value is a Tensor with shape [2,1], which can be broadcast to [2,3]
>>> a
Tensor(shape=[2, 3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[[1. , 2. , 10., 5. ],
         [1. , 2. , 10., 5. ],
         [1. , 2. , 10., 5. ]],

        [[1. , 2. , 10., 5. ],
         [1. , 2. , 10., 5. ],
         [1. , 2. , 10., 5. ]]])

>>> a[:,:,3] = paddle.full([2,4], 5) # value is a Tensor with shape [2,4], which cannot be broadcast to [2,3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.8/dist-packages/paddle/base/dygraph/tensor_patch_methods.py", line 996, in __setitem__
    return self._setitem_dygraph(item, value)
ValueError: (InvalidArgument) The shape of tensor assigned value must match the shape of target shape: [2, 4], but now shape is [2, 3].
```

### 5.2 静态图下赋值请使用 API paddle.static.setitem
由于赋值是一个原地(in-place)操作，这在静态图组网时通常会违背静态单赋值（Static Single-Assignment, SSA）原则，导致反向计算时梯度不正确。因此，在静态图下，飞桨禁用了 Tensor 的`__setitem__`调用，并提供作为替代的 out-place API `paddle.static.setitem(x, index, value)`。这个 API 将会返回赋值后的结果，并独立于`x`。

在动态图下，仍然可以直接调用`__setitem__`，飞桨底层同时提供了对应的动转静策略以保证该场景在动转静时的正确性。

```python
>>> paddle.enable_static()
>>> a = paddle.ones((2,3,4))
>>> a[0] = 1
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.8/dist-packages/paddle/base/framework.py", line 2585, in __setitem__
    raise RuntimeError(
RuntimeError: In static mode, the __setitem__ (looks like: x[indices] = values) should not be used. Please use x = paddle.static.setitem(x, indices, values)

>>> b = paddle.static.setitem(a, 0, 1) # use paddle.static.setitem instead of __setitem__ in static mode
>>> b
var set_value_0.tmp_0 : LOD_TENSOR.shape(2, 3, 4).dtype(float32).stop_gradient(True)
```

### 5.3 不同数据类型时的行为
在飞桨中，当出现待索引 Tensor `x`和值`value`的数据类型不同时，会以**待索引 Tensor 的数据类型为准**。因此，在`value`的数据类型高于`x`时，可能出现数据截断，在实际使用中应尽量避免这种情况。

```python
>>> a = paddle.ones((2,3,4), dtype='int32')
>>> a[0] = 2.5  # the value is truncated since float is casted to int
>>> a
Tensor(shape=[2, 3, 4], dtype=int32, place=Place(cpu), stop_gradient=True,
       [[[2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2]],

        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]])

>>> a = paddle.full([2,3], 1.25)
>>> a[0] = 10  # the int value is casted to float
>>> a
Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[10.       , 10.       , 10.       ],
        [1.25000000, 1.25000000, 1.25000000]])
```
