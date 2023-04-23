# Introduction to Tensor

## Introduction: Concept of Tensor

PaddlePaddle(Hereinafter referred to as Paddle) is the same as other Deep Learning Framework, it use **Tensor** to
representing data.

**Tensor** can be regarded as multi-dimensional array, which can have as many diemensions as it want. Different **Tensor** may have different data types(dtype) and shapes.

The dtypes of all elements in the same Tensor are the same. If you are familiar with [Numpy](https://numpy.org/doc/stable/user/quickstart.html#the-basics), **Tensor** is similar to the **Numpy array**.

## Chapter1. Creation of Tensor

Paddle provides several methods to create a **Tensor**, such as creating by specifying data list, by specifying shape, by specifying interval.

### 1. create by specifying data list

By specifying data list, you can create tensor of any diemensions(sometimes called "axes"). For example:
```python
# create **1-D Tensor** like matrix
import paddle # The following sample code has imported the paddle module by default
ndim_1_tensor = paddle.to_tensor([2.0, 3.0, 4.0])
print(ndim_1_tensor)
```
```text
Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [2., 3., 4.])
```

Specifically, if you input only a scalar data (for example, float/int/bool), then a **Tensor** whose shape is [1] will be created.
```python
paddle.to_tensor(2)
paddle.to_tensor([2])
```
The above two are completely the same, Tensor shape is [1] , and the output is:
```text
Tensor(shape=[1], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [2])
```

```python
# create **2-D Tensor** like matrix
ndim_2_tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(ndim_2_tensor)
```
```text
Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[1., 2., 3.],
        [4., 5., 6.]])
```

```python
# create multidimensional Tensor
ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(ndim_3_tensor)
```
```text
Tensor(shape=[2, 2, 5], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [[[1 , 2 , 3 , 4 , 5 ],
         [6 , 7 , 8 , 9 , 10]],

        [[11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20]]])
```
The visual representation of the **Tensor* above is:

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Tensor_2.0.png?raw=true" width="800" ></center>
<br><center>Figure1. Visual representation of Tensor with different dimensions</center>

**Tensor** must be "rectangular" -- that is, along each dimension, the number of elements must be equal. For example:
```
ndim_2_tensor = paddle.to_tensor([[1.0, 2.0],
                                  [4.0, 5.0, 6.0]])
```
An exception will be thrown in this case:
```text
ValueError:
        Failed to convert input data to a regular ndarray :
         - Usually this means the input data contains nested lists with different lengths.
```

### 2. create by specifying shape
If you want to create a **Tensor** of specific shape, you can use API below:

```python
paddle.zeros([m, n])             # create Tensor of all elements: 0, Shape: [m, n]
paddle.ones([m, n])              # create Tensor of all elements: 1, Shape: [m, n]
paddle.full([m, n], 10)          # create Tensor of all elements: 10, Shape: [m, n]
```
For example, the output of `paddle.ones([2,3])` is：
```text
Tensor(shape=[2, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[1., 1., 1.],
        [1., 1., 1.]])
```
### 3. create by specifying interval

If you want to create a **Tensor** of specific interval, you can use API below:

```python
paddle.arange(start, end, step)  # create Tensor within interval [start, end) evenly separated by step
paddle.linspace(start, stop, num) # create Tensor within interval [start, stop) evenly separated by elements number
```
For example, the output of `paddle.arange(start=1, end=5, step=1)` is：
```text
Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [1, 2, 3, 4])
```

## Chapter2. Attributes of Tensor

### 2.1 shape of Tensor

The shape of **Tensor** can be get by **Tensor.shape**. shape is an important attribute of **Tensor**, and the following are related concepts:

1. shape: Describes the number of elements on each of the tensor's dimensions.
2. ndim: The number of tensor's dimensions. For example, the ndim of vector is 1, the ndim of matrix is 2.
3. axis or dimension: A particular dimension of a tensor.
4. size: The number of all elements in the tensor.

Create a 4-D **Tensor**, and visualize it to represents the relationship between the above concepts.
```python
ndim_4_tensor = paddle.ones([2, 3, 4, 5])
```

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Axis_2.0.png?raw=true" width="800" ></center>
<br><center>Figure2. The relationship between Tensor shape, axis, dimension and ndim</center>

```python
print("Data Type of every element:", ndim_4_tensor.dtype)
print("Number of dimensions:", ndim_4_tensor.ndim)
print("Shape of tensor:", ndim_4_tensor.shape)
print("Elements number along axis 0 of tensor:", ndim_4_tensor.shape[0])
print("Elements number along the last axis of tensor:", ndim_4_tensor.shape[-1])
```
```text
Data Type of every element: paddle.float32
Number of dimensions: 4
Shape of Tensor: [2, 3, 4, 5]
Elements number along axis 0 of Tensor: 2
Elements number along the last axis of Tensor: 5
```

Manipulating shape of Tensor is important in programming, Paddle provides reshape API to manipulate the shape of Tensor:
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

There are some tricks for specifying a new shape:

**1.** -1 indicates that the value of this dimension is inferred from the total number of elements and the other dimension of Tensor. Therefore, there is one and only one that can be set to -1.

**2.** 0 means that the actual dimension is copied from the corresponding dimension of Tensor, so the index value of 0 in shape can't exceed the ndim of the Tensor.

For example:
```text
origin:[3, 2, 5] reshape:[3, 10]      actual: [3, 10]
origin:[3, 2, 5] reshape:[-1]         actual: [30]
origin:[3, 2, 5] reshape:[0, 5, -1]   actual: [3, 5, 2]
```

When reshape is [-1], Tensor will be flattened to 1-D according to its layout in computer memory.
```python
print("Tensor flattened to Vector:", paddle.reshape(ndim_3_tensor, [-1]).numpy())
```
```text
Tensor flattened to Vector: [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
```

### 2.2 dtype of Tensor

data type of **Tensor**, which can be get from Tensor.dtype, it support 'bool', 'float16', 'float32', 'float64','uint8', 'int8', 'int16', 'int32', 'int64'.

* If create Tensor from Python elements, the data type can be specified by dtype. Otherwise:

    * For python integer data, it will create int64 Tensor
    * For python floats number, it will create float32 Tensor by default. You can change default dtype by set_default_type.

* If create Tensor from Numpy array, the data type remains the same with origin dtype.

```python
print("Tensor dtype from Python integers:", paddle.to_tensor(1).dtype)
print("Tensor dtype from Python floating point:", paddle.to_tensor(1.0).dtype)
```
```text
Tensor dtype from Python integers: paddle.int64
Tensor dtype from Python floating point: paddle.float32
```

**Tensor** supports not only floats and ints but also complex numbers data, If input complex number data, the dtype of **Tensor** is ``complex64`` or ``complex128`` :

```python
ndim_2_tensor = paddle.to_tensor([[(1+1j), (2+2j)],
                                  [(3+3j), (4+4j)]])
print(ndim_2_tensor)
```

```text
Tensor(shape=[2, 2], dtype=complex64, place=Place(gpu:0), stop_gradient=True,
       [[(1+1j), (2+2j)],
        [(3+3j), (4+4j)]])
```


Paddle provides **cast** API to change the dtype:
```python
float32_tensor = paddle.to_tensor(1.0)

float64_tensor = paddle.cast(float32_tensor, dtype='float64')
print("Tensor after cast to float64:", float64_tensor.dtype)

int64_tensor = paddle.cast(float32_tensor, dtype='int64')
print("Tensor after cast to int64:", int64_tensor.dtype)
```
```text
Tensor after cast to float64: paddle.float64
Tensor after cast to int64: paddle.int64
```

### 2.3 place of Tensor

Device can be specified when creating a tensor. There are three kinds of to choose from: CPU/GPU/Pinned memory.There is higher read and write efficiency between Pinned memory with GPU. In addition, Pinned memory supports asynchronous data copy, which will further improve the performance of network. The disadvantage is that allocating too much Pinned memory may reduce the performance of the host. Because it reduces the pageable memory which is used to store virtual memory data. When place is not specified, the default place of Tensor is consistent with the version of Paddle, for example, the place is GPU by default if you installed paddlepaddle-gpu.

The following example creates Tensors on CPU, GPU, and pinned memory respectively, and uses `Tensor.place` to view the device place where the Tensor is located:

* **Create Tensor on CPU**:
```python
cpu_Tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
print(cpu_Tensor.place)
```

```text
Place(cpu)
```

* **Create Tensor on GPU**:
```python
gpu_Tensor = paddle.to_tensor(1, place=paddle.CUDAPlace(0))
print(gpu_Tensor.place) # the output shows that the Tensor is On the 0th graphics card of the GPU device
```

```text
Place(gpu:0)
```

* **Create Tensor on pinned memory**:
```python
pin_memory_Tensor = paddle.to_tensor(1, place=paddle.CUDAPinnedPlace())
print(pin_memory_Tensor.place)
```

```text
Place(gpu_pinned)
```
### name of Tensor

name of Tensor is its unique identifier, which is a Python string, and it can be get by ``Tensor.name``. By default, Paddle will customize a unique name when creating a Tensor.

```python
print("Tensor name:", paddle.to_tensor(1).name)
```
```text
Tensor name: generated_tensor_0
```

## Chapter3. Method of Tensor

### 3.1 Index and slice

You can easily access or modify Tensors by indexing or slicing. Paddle follows standard Python indexing rules, similar to [Indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings) and the basic rules for NumPy indexing. It has the following features:

1. Indexing a Tensor based on the subscript 0-n. A negative subscript means counting backwards from the end.
2. Slicing a Tensor base on separating parameters `start:stop:step` by colons `:`, and `start`, `stop` and `step` can be default.

#### Access Tensor
For **1-D Tensor**, there is only single-axis indexing or slicing:
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

For 2-D **Tensor** or above, there is multi-axis indexing or slicing:
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

The first element of index or slice is corresponds to axis 0, the second is corresponds to axis 1, and so on. If no index is specified on an axis, the default is `:` . For example:
```python
ndim_2_tensor[1]
ndim_2_tensor[1, :]
```
The result of these two operations are exactly the same.

```text
Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [4, 5, 6, 7])
```

#### Modify Tensor

> **Warning:**
>
> Please be careful to modify a Tensor through index or slice. It will **inplace** modify the value of Tensor, and the original value will not be saved. If the modified Tensor participates in the gradient calculation, only the modified value will be used, which may introduce risks to the gradient calculation. Paddle will detect and report errors in risky operations later.

Similar to accessing a Tensor, modifying a Tensor by indexing or slicing can be on a single or multiple axes. In addition, it supports assigning multiple types of data to a Tensor. The supported data types are `int`, `float`,  `numpy.ndarray` and `Tensor`.

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

In addition, Paddle provides rich Tensor operating APIs, including Mathematical operations, logical operations, linear algebra and so on. The total number is more than 100 kinds. For example:

```python
x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float64")
y = paddle.to_tensor([[5.5, 6.6], [7.7, 8.8]], dtype="float64")

print(paddle.add(x, y), "\n") # 1st method
print(x.add(y), "\n") # 2nd method
```

```text
Tensor(shape=[2, 2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       [[6.60000000 , 8.80000000 ],
        [11.        , 13.20000000]])

Tensor(shape=[2, 2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       [[6.60000000 , 8.80000000 ],
        [11.        , 13.20000000]])
```

It can be seen that Tensor class method has the same result with Paddle API. And since the Tensor class method is more convenient to invoke, the following is an introduction to common **Tensor** operations from the perspective of **Tensor class member functions**.

### 3.2 mathematical operations
```python
x.abs()                       #absolute value
x.ceil()                      #round up to an integer
x.floor()                     #round down to an integer
x.round()                     #round to an integer
x.exp()                       #calculate exponents of the natural constant of each element
x.log()                       #calculate natural logarithm of each element
x.reciprocal()                #reciprocal of each element
x.square()                    #calculate square of each element
x.sqrt()                      #calculate sqrt of each element
x.sin()                       #calculate the sine of each element
x.cos()                       #calculate the cosine of each element
x.add(y)                      #add element by element
x.subtract(y)                 #subtract element by element
x.multiply(y)                 #multiply element by element
x.divide(y)                   #divide element by element
x.mod(y)                      #mod element by element
x.pow(y)                      #pow element by element
x.max()                       #the maximum element on specific axis
x.min()                       #the minimum element on specific axis
x.prod()                      #multiply all elements on specific axis
x.sum()                       #sum of all elements on specific axis
```

Paddle overwrite the magic functions related to Python mathematical operations. The following operations have the same result as above.
```text
x + y  -> x.add(y)            #add element by element
x - y  -> x.subtract(y)       #subtract element by element
x * y  -> x.multiply(y)       #multiply element by element
x / y  -> x.divide(y)         #divide element by element
x % y  -> x.mod(y)            #mod element by element
x ** y -> x.pow(y)            #pow element by element
```

### 3.3 logical operations
```python
x.isfinite()                  #Judge whether the element in tensor is finite number
x.equal_all(y)                #Judge whether all elements of two tensor are equal
x.equal(y)                    #judge whether each element of two tensor is equal
x.not_equal(y)                #judge whether each element of two tensor is not equal
x.less_than(y)                #judge whether each element of tensor x is less than corresponding element of tensor y
x.less_equal(y)               #judge whether each element of tensor x is less than or equal to element of tensor y
x.greater_than(y)             #judge whether each element of tensor x is greater than element of tensor y
x.greater_equal(y)            #judge whether each element of tensor x is greater than or equal to element of tensor y
x.allclose(y)                 #judge whether all elements of tensor x is close to all elements of tensor y
```

Paddle overwrite the magic functions related to Python logical operations. The following operations have the same result as above.
```text
x == y  -> x.equal(y)         #judge whether each element of two tensor is equal
x != y  -> x.not_equal(y)     #judge whether each element of two tensor is not equal
x < y   -> x.less_than(y)     #judge whether each element of tensor x is less than corresponding
x <= y  -> x.less_equal(y)    #judge whether each element of tensor x is less than or equal to element of tensor y
x > y   -> x.greater_than(y)  #judge whether each element of tensor x is greater than element of tensor y
x >= y  -> x.greater_equal(y) #judge whether each element of tensor x is greater than or equal to element of tensor y
```

The following operations are targeted at bool Tensor only:
```python
x.logical_and(y)              #logic and operation for two bool tensor
x.logical_or(y)               #logic or operation for two bool tensor
x.logical_xor(y)              #logic xor operation for two bool tensor
x.logical_not(y)              #logic not operation for two bool tensor
```

### 3.4 linear algebra
```python
x.t()                         #matrix transpose
x.transpose([1, 0])           #swap axis 0 with axis 1
x.norm('fro')                 #Frobenius Norm of matrix
x.dist(y, p=2)                #The 2 norm of (x-y)
x.matmul(y)                   #Matrix multiplication
```
> **Warning:**
>
> The API in Paddle is divided into in-place operations and non-in-place operations. The in-place  operation saves the operation result on the original **Tensor**, while the non-in-place operation  returns a new **Tensor** to represent the operation result instead of modifying the original **Tensor**. After Paddle 2.1, some APIs have corresponding in-place operation versions, adding `_` after the API to indicate, for example, `x.add(y)` is a non-in-place operation, and `x.add_(y)` is a in-place operation.

For more API related to Tensor operations, please refer to [class paddle.Tensor]((../../../api/paddle/Tensor_cn.html))

## Chapter4. Convert between Tensor and Numpy array

### 4.1 Convert Tensor to numpy array
Convert **Tensor** to **Numpy array** through the Tensor.numpy() method:
```python
tensor_to_convert = paddle.to_tensor([1.,2.])
tensor_to_convert.numpy()
```
```text
array([1., 2.], dtype=float32)
```

### 4.2 Convert numpy array to Tensor

Convert **Numpy array** to **Tensor** through paddle.to_tensor() method:
```python
tensor_temp = paddle.to_tensor(np.array([1.0, 2.0]))
print(tensor_temp)
```
```text
Tensor(shape=[2], dtype=float64, place=Place(gpu:0), stop_gradient=True,
       [1., 2.])
```
The created **Tensor** will have the same shape and dtype with the original Numpy array.

## Chapter5. Broadcasting of Tensor

PaddlePaddle provides broadcasting semantics in some APIs like other deep learning frameworks, which allows using tensors with different shapes while operating.
In General, broadcast is the rule how the smaller tensor is “broadcast” across the larger tsnsor so that they have same shapes.
Note that no copies happened while broadcasting.

In PaddlePaddle, tensors are broadcastable when following rulrs hold(ref [Numpy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting)):

1. there should be at least one dimention in each tensor
2. when comparing their shapes element-wise from backward to forward, two dimensions are compatible when
   they are equal, or one of them is 1, or one of them does not exist.

For example:

```python
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 4))
# Two tensor have some shapes are broadcastable
z = x + y
print(z.shape)
# [2, 3, 4]

x = paddle.ones((2, 3, 1, 5))
y = paddle.ones((3, 4, 1))

# compare from backward to forward：
# 1st step：y's dimention is 1
# 2nd step：x's dimention is 1
# 3rd step：two dimentions are the same
# 4st step：y's dimention does not exist
# So, x and y are broadcastable
z = x + y
print(z.shape)
# [2, 3, 4, 5]

# In Compare
x = paddle.ones((2, 3, 4))
y = paddle.ones((2, 3, 6))
# x and y are not broadcastable because in first step form tail, x's dimention 4 is not equal to y's dimention 6
# z = x, y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.
```

Now you know in what condition two tensors are broadcastable, how to calculate the resulting tensor's size follows the rules:

1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
2. Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

For example:

```python
x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 1))
z = x + y
print(z.shape)
# z'shape: [2, 3, 4]

x = paddle.ones((2, 1, 4))
y = paddle.ones((3, 2))
# z = x + y
# ValueError: (InvalidArgument) Broadcast dimension mismatch.
```
