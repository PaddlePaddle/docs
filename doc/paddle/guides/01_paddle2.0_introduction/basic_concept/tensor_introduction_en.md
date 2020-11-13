# Introduction to Tensor

PaddlePaddle(Hereinafter referred to as Paddle) is the same as other Deep learning Framework, it use **Tensor** to
representing data.

**Tensor** can be regarded as multi-dimensional array, which can have as many diemensions as it want. Different **Tensor** can have different data types(dtype) and shape.

The dtypes of all elements in the same Tensor are the same. If you are familiar with [Numpy](https://www.paddlepaddle.org.cn/tutorials/projectdetail/590690), **Tensor** is similar to the **Numpy array**.

## Creation of Tensor

Firstly, let we create a **Tensor**:

### 1. create **1-D Tensor** like vector, whose rank is 1
```python
# The Tensor data type can be specified by dtype, otherwise, float32 Tensor will be created
rank_1_tensor = paddle.to_tensor([2.0, 3.0, 4.0], dtype='float64')
print(rank_1_tensor)
```
```text
Tensor(shape=[3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
       [2., 3., 4.])
```

Specifically, if you imput only a scalar data (for example, float/int/bool), then a **Tensor** whose shape is [1]will be created.
```python
paddle.to_tensor(2)
paddle.to_tensor([2])
```
The above two are completely the same, Tensor shape is [1]:
```text
Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [2])
```

### 2. create **2-D Tensor** like matrix, whose rank is 2
```python
rank_2_tensor = paddle.to_tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]])
print(rank_2_tensor)
```
```text
Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [[1., 2., 3.],
        [4., 5., 6.]])
```

### 3. Similarly, you can create multidimensional Tensor whose rank is 3, 4... N
```
# There can be an arbitrary number of axes (sometimes called "dimensions")
rank_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]]])
print(rank_3_tensor)
```
```text
Tensor(shape=[2, 2, 5], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [[[1, 2, 3, 4, 5],
         [ 6,  7,  8,  9, 10]],

        [[11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20]]])
```
The visual representation of the **Tensor* above is:

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Tensor_2.0.png?raw=true" width="800" ></center>
<br><center>Figure1. Visual representation of Tensor with different ranks</center>


You can convert **Tensor** to Numpy array easily Tensor.numpy() method.
```python
print(rank_2_tensor.numpy())
```
```text
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)
```

**Tensor supports not only floats and ints but also complex Numbers data:
```python
rank_2_complex_tensor = paddle.to_tensor([[1+1j, 2+2j],
                                          [3+3j, 4+4j]])
```
```text
CompleTensor[real]: generated_tensor_0.real
  - place: CUDAPlace(0)
  - shape: [2, 2]
  - layout: NCHW
  - dtype: float
  - data: [1 2 3 4]
CompleTensor[imag]: generated_tensor_0.real
  - place: CUDAPlace(0)
  - shape: [2, 2]
  - layout: NCHW
  - dtype: float
  - data: [1 2 3 4]
```
If the input data contains complex Number, a **ComplexTensor** is automatically created. **ComplexTensor** is a special data structure in Paddle. **ComplexTensor** consists of two **Tensor**, one is real part and the other is  imaginary part. **ComplexTensor** can be visualized as follows:

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/ComplexTensor_2.0.png?raw=true" width="800" ></center>
<br><center>Figure2. Visual representation of ComplexTensor</center>

**Tensor** must be "rectangular" -- that is, along each axis, every element is the same size. For example:
```
rank_2_tensor = paddle.to_tensor([[1.0, 2.0],
                                  [4.0, 5.0, 6.0]])
```
An exception will be thrown in this case:
```text
ValueError:
    Faild to convert input data to a regular ndarray :
     - Usually this means the input data contains nested lists with different lengths.
```

The way to create **Tensor** from Python data is described above. We can also create **Tensor**
from numpy array:
```python
rank_1_tensor = paddle.to_tensor(numpy.array([1.0, 2.0]))

rank_2_tensor = paddle.to_tensor(numpy.array([[1.0, 2.0],
                                              [3.0, 4.0]]))

rank_3_tensor = paddle.to_tensor(numpy.random.rand(3, 2))
```
The created **Tensor** will have the same shape and dtype with the original Numpy array.

If you want to create a **Tensor** with specific size, Paddle also provide these API:
```text
paddle.zeros([m, n])             # All elements: 0, Shape: [m, n]
paddle.ones([m, n])              # All elements: 1, Shape: [m, n]
paddle.full([m, n], 10)          # All elements: 10, Shape: [m, n]
paddle.arrange(start, end, 2) # Elements: from start to end, step size is 2
paddle.linspace(start, end, 10) # Elements: from start to end, num of elementwise is 10
```


## Shape of Tensor

### Basic Concept

The shape of **Tensor** can be get by **Tensor.shape**. shape is an important attribute of **Tensor**, and the following are related concepts:

1. shape: Describes the number of elements on each of the tensor's dimensions.
2. rank: The number of tensor's dimensions. For example, the rank of vector is 1, the rank of matrix is 2.
3. axis or dimension: A particular dimension of a tensor.
4. size: The number of all elements in the tensor.

Let we create a 4-D **Tensor**, and visualize it to represents the relationship between the above concepts.
```python
rank_4_tensor = paddle.ones([2, 3, 4, 5])
```

<center><img src="https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/Axis_2.0.png?raw=true" width="800" ></center>
<br><center>Figure3. The relationship between Tensor shape, axis, dimension and rank</center>

```python
print("Data Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements number along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements number along the last axis of tensor:", rank_4_tensor.shape[-1])
```
```text
Data Type of every element: VarType.FP32
Number of dimensions: 4
Shape of tensor: [2, 3, 4, 5]
Elements number along axis 0 of tensor: 2
Elements number along the last axis of tensor: 5
```

### Manipulating Shape

Manipulating shape of Tensor is important in programming.
```python
rank_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20]],
                                  [[21, 22, 23, 24, 25],
                                   [26, 27, 28, 29, 30]]])
print("the shape of rank_3_tensor:", rank_3_tensor.shape)
```
```text
the shape of rank_3_tensor: [3, 2, 5]
```

Paddle provides reshape API to manipulate the shape of Tensor:
```python
rank_3_tensor = paddle.reshape(rank_3_tensor, [2, 5, 3])
print("After reshape:", rank_3_tensor.shape)
```
```text
After reshape: [2, 5, 3]
```

There are some tricks for specifying a new shape:

1. -1 indicates that the value of this dimension is inferred from the total number of elements and the other dimension of Tensor. Therefore, there is one and only one that can be set to -1.
2. 0 means that the actual dimension is copied from the corresponding dimension of Tensor, so the index value of 0 in shape can't exceed the rank of X.

For example:
```text
origin:[3, 2, 5] reshape:[3, 10]     actual: [3, 10]
origin:[3, 2, 5] reshape:[-1]         actual: [30]
origin:[3, 2, 5] reshape:[0, 5, -1] actual: [3, 5, 2]
```

If you flatten a tensor by reshape to -1, you can see what order it is laid out in memory.
```python
print("Tensor flattened to Vector:", paddle.reshape(rank_3_tensor, [-1]).numpy())
```
```text
Tensor flattened to Vector: [1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
```


## Other attributes of Tensor

### dtype of Tensor

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
Tensor dtype from Python integers: VarType.INT64
Tensor dtype from Python floating point: VarType.FP32
```

Paddle provide **cast** API to change the dtype:
```python
float32_tensor = paddle.to_tensor(1.0)

float64_tensor = paddle.cast(float32_tensor, dtype='float64')
print("Tensor after cast to float64:", float64_tensor.dtype)

int64_tensor = paddle.cast(float32_tensor, dtype='int64')
print("Tensor after cast to int64:", int64_tensor.dthpe)
```
```text
Tensor after cast to float64: VarType.FP64
Tensor after cast to int64: VarType.INT64
```

### place of Tensor

Device can be specified when creating a tensor. There are three kinds of to choose from: CPU/GPU/Pinned memory.
There is higher read and write efficiency between Pinned memory with GPU. In addition, Pinned memory supports asynchronous data copy, which will further improve the performance of network. The disadvantage is that allocating too much Pinned memory may reduce the performance of the host. Because it reduces the pageable memory which is used to store virtual memory data.

* **Create Tensor on GPU**:
```python
cpu_tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
print(cpu_tensor)
```

```text
Tensor(shape=[1], dtype=int64, place=CPUPlace, stop_gradient=True,
       [1])
```

* **Create Tensor on CPU**:
```python
gpu_tensor = paddle.to_tensor(1, place=paddle.CUDAPlace(0))
print(gpu_tensor)
```

```text
Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [1])
```

* **Create Tensor on pinned memory**:
```python
pin_memory_tensor = paddle.to_tensor(1, place=paddle.CUDAPinnedPlace())
print(pin_memory_tensor)
```
```text
Tensor(shape=[1], dtype=int64, place=CUDAPinnedPlace, stop_gradient=True,
       [1])

```
### name of Tensor

name of Tensor is its unique identifier, which is a Python string, and it can be get by ``Tensor.name``. By default, Paddle will customize a unique name when creating a Tensor.

```python
print("Tensor name:", paddle.to_tensor(1).name)
```
```text
Tensor name: generated_tensor_0
```


## Method of Tensor

### Index and slice

You can easily access or modify Tensors by indexing or slicing. Paddle follows standard Python indexing rules, similar to [Indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings) and the basic rules for NumPy indexing. It has the following features:

1. Indexing a Tensor based on the subscript 0-n. A negative subscript means counting backwards from the end.
2. Slicing a Tensor base on separating parameters `start:stop:step` by colons `:`, and `start`, `stop` and `step` can be default.

#### Access Tensor
For **1-D Tensor**, there is only single-axis indexing or slicing:
```python
rank_1_tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", rank_1_tensor.numpy())

print("First element:", rank_1_tensor[0].numpy())
print("Last element:", rank_1_tensor[-1].numpy())
print("All element:", rank_1_tensor[:].numpy())
print("Before 3:", rank_1_tensor[:3].numpy())
print("From 6 to the end:", rank_1_tensor[6:].numpy())
print("From 3 to 6:", rank_1_tensor[3:6].numpy())
print("Interval of 3:", rank_1_tensor[::3].numpy())
print("Reverse:", rank_1_tensor[::-1].numpy())
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
rank_2_tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print("Origin Tensor:", rank_2_tensor.numpy())

print("First row:", rank_2_tensor[0].numpy())
print("First row:", rank_2_tensor[0, :].numpy())
print("First column:", rank_2_tensor[:, 0].numpy())
print("Last column:", rank_2_tensor[:, -1].numpy())
print("All element:", rank_2_tensor[:].numpy())
print("First row and second column:", rank_2_tensor[0, 1].numpy())
```
```text
Origin Tensor: array([[ 0  1  2  3]
                      [ 4  5  6  7]
                      [ 8  9 10 11]], dtype=int64)
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
```
rank_3_tensor[1]
rank_3_tensor[1, :]
rank_3_tensor[1, :, :]
```
These three are exactly the same.

#### Modify Tensor

> **Warning:**
>
> Please be careful to modify a Tensor through index or slice. It will **inplace** modify the value of Tensor, and the original value will not be saved. If the modified Tensor participates in the gradient calculation, only the modified value will be used, which may introduce risks to the gradient calculation. Paddle will detect and report errors in risky operations later.

Similar to accessing a Tensor, modifying a Tensor by indexing or slicing can be on a single or multiple axes. In addition, it supports assigning multiple types of data to a Tensor. The supported data types are `int`, `float`,  `numpy.ndarray` and `Tensor`.

```python
import paddle
import numpy as np

x = paddle.to_tensor(np.ones((2, 3)).astype(np.float32)) # [[1,1,1], [1,1,1]]

x[0] = 0                      # x : [[0, 0, 0], [1 ,1, 1]]        id(x) = 4433705584
x[0:1] = 2.1                  # x : [[2.1, 2.1, 2.1], [1 ,1, 1]]  id(x) = 4433705584
x[...] = 3                    # x : [[3, 3, 3], [3, 3, 3]]        id(x) = 4433705584

x[0:1] = np.array([1,2,3])    # x : [[1, 2, 3], [3, 3, 3]]        id(x) = 4433705584
x[1] = paddle.ones([3])       # x : [[1, 2, 3], [1,1,1]]          id(x) = 4433705584
```

---

In addition, Paddle provides rich Tensor operating APIs, including mathematical operators, logical operators, linear algebra operators and so on. The total number is more than 100 kinds. For example:

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

It can be seen that Tensor class method has the same result with Paddle API. And the Tensor class method is more convenient to invoke.


### mathematical operators
```python
x.abs()                       #absolute value
x.ceil()                      #round up to an integer
x.floor()                     #round down to an integer
x.exp()                       #Calculate exponents of the natural constant of each element
x.log()                       #Calculate natural logarithm of each element
x.reciprocal()                #reciprocal
x.square()                    #Calculate square of each element
x.sqrt()                      #Calculate sqrt of each element
x.sum()                       #Calculate the sum of all elements
x.asin()                      #Calculate the arcsine of each element
x.add(y)                      #add element by element
x.add(-y)                     #minus element by element
x.multiply(y)                 #multiply element by element
x.divide(y)                   #divide element by element
x.floor_divide(y)             #divide exactly element by element
x.remainder(y)                #mod element by element
x.pow(y)                      #pow element by element
x.reduce_max()                #max value on specific axis
x.reduce_min()                #min value on specific axis
x.reduce_prod()               #multiply of all elements on specific axis
x.reduce_sum()                #sum of all elements on specific axis
```

Paddle overwrite the magic functions related to Python mathematical operations. Like this:
```text
x + y  -> x.add(y)  
x - y  -> x.add(-y)  
x * y  -> x.multiply(y)  
x / y  -> x.divide(y)  
x // y -> x.floor_divide(y)
x % y  -> x.remainder(y)
x ** y -> x.pow(y)
```

### logical operators
```python
x.is_empty()                  #Judge whether tensor is empty
x.isfinite()                  #Judge whether the element in tensor is finite number
x.euqal_all(y)                #Judge whether all elements of two tensor are equal
x.euqal(y)                    #judge whether each element of two tensor is equal
x.not_equal(y)                #judge whether each element of two tensor is not equal
x.less_than(y)                #judge whether each element of tensor x is less than corresponding element of tensor y
x.less_equal(y)               #judge whether each element of tensor x is less than or equal to element of tensor y
x.greater_than(y)             #judge whether each element of tensor x is greater than element of tensor y
x.greater_equal(y)            #judge whether each element of tensor x is greater than or equal to element of tensor y
```

Paddle overwrite the magic functions related to Python logical operations. Like this:
```text
x == y  -> x.euqal(y)  
x != y  -> x.not_equal(y)  
x < y   -> x.less_than(y)  
x <= y  -> x.less_equal(y)  
x > y   -> x.greater_than(y)  
x >= y  -> x.greater_equal(y)
```

The following operations are targeted at bool Tensor only:
```python
x.reduce_all(y)               #Judge whether a bool tensor is True for all elements
x.reduce_any(y)               #Judge whether a bool tensor exists at least one element is True
x.logical_and(y)              #logic and operation for two bool tensor
x.logical_or(y)               #logic or operation for two bool tensor
x.logical_xor(y)              #logic xor operation for two bool tensor
x.logical_not(y)              #logic not operation for two bool tensor
```

### linear algebra operators
```python
x.cholesky()                  #cholesky decomposition of a matrix
x.t()                         #matrix transpose
x.transpose([1, 0])           #swap axis 0 with axis 1
x.norm('pro')                 #Frobenius Norm of matrix
x.dist(y, p=2)                #The 2 norm of (x-y)
x.matmul(y)                   #Matrix multiplication
```
It should be noted that the class method of Tensor are non-inplace operations. It means, ``x.And dd(y)`` will not operate directly on Tensor x, but return a new Tensor to represent the results.

For more API related to Tensor operations, please refer to [class paddle.Tensor]((../../../api/paddle/tensor/creation/Tensor_cn.html))
