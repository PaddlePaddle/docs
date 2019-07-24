.. _cn_api_fluid_LoDTensor:

LoDTensor
-------------------------------

.. py:class:: paddle.fluid.LoDTensor


LoDTensor是一个具有LoD信息的张量(Tensor)

``np.array(lod_tensor)`` 可以将LoDTensor转换为numpy array。

``lod_tensor.lod()`` 可以获得LoD信息。

LoD是多层序列（Level of Details）的缩写，通常用于不同长度的序列。如果您不需要了解LoD信息，可以跳过下面的注解。

举例:

X 为 LoDTensor，它包含两个逻辑子序列。第一个长度是2，第二个长度是3。

从Lod中可以计算出X的第一维度为5， 因为5=2+3。在X中的每个序列中的每个元素有2列，因此X的shape为[5,2]。

::

  x.lod  =  [[2, 3]] 
  
  x.data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

  x.shape = [5, 2]


LoD可以有多个level(例如，一个段落可以有多个句子，一个句子可以有多个单词)。下面的例子中，Y为LoDTensor ，lod_level为2。表示有2个逻辑序列，第一个逻辑序列的长度是2(有2个子序列)，第二个逻辑序列的长度是1。第一个逻辑序列的两个子序列长度分别为2和2。第二个序列的子序列的长度是3。


::
  
  y.lod = [[2 1], [2 2 3]]

  y.shape = [2+2+3, ...]

**示例代码**

.. code-block:: python

      import paddle.fluid as fluid
     
      t = fluid.LoDTensor()

.. note::

  在上面的描述中，LoD是基于长度的。在paddle内部实现中，lod是基于偏移的。因此,在内部,y.lod表示为[[0,2,3]，[0,2,4,7]](基于长度的Lod表示为为[[2-0,3-2]，[2-0,4-2,7-4]])。

  可以将LoD理解为recursive_sequence_length（递归序列长度）。此时，LoD必须是基于长度的。由于历史原因，当LoD在API中被称为lod时，它可能是基于偏移的。用户应该注意。




.. py:method:: has_valid_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor) → bool

检查LoDTensor的lod值的正确性。

返回:    是否带有正确的lod值

返回类型:    out (bool)

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])
            print(t.has_valid_recursive_sequence_lengths()) # True

.. py:method::  lod(self: paddle.fluid.core_avx.LoDTensor) → List[List[int]]

得到LoD Tensor的LoD。

返回：LoD Tensor的LoD。

返回类型：out（List [List [int]]）

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_lod([[0, 2, 5]])
            print(t.lod()) # [[0, 2, 5]]

.. py:method:: recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor) → List[List[int]]

得到与LoD对应的LoDTensor的序列长度。

返回：LoD对应的一至多个序列长度。

返回类型：out（List [List [int]）

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])
            print(t.recursive_sequence_lengths()) # [[2, 3]]


.. py:method::  set(*args, **kwargs)
    
重载函数

1. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[float32], arg1: paddle::platform::CPUPlace) -> None

2. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[int32], arg1: paddle::platform::CPUPlace) -> None

3. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[float64], arg1: paddle::platform::CPUPlace) -> None

4. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[int64], arg1: paddle::platform::CPUPlace) -> None

5. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[bool], arg1: paddle::platform::CPUPlace) -> None

6. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[uint16], arg1: paddle::platform::CPUPlace) -> None

7. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[uint8], arg1: paddle::platform::CPUPlace) -> None

8. set(self: paddle.fluid.core_avx.Tensor, arg0: numpy.ndarray[int8], arg1: paddle::platform::CPUPlace) -> None

.. py:method::  set_lod(self: paddle.fluid.core_avx.LoDTensor, lod: List[List[int]]) → None

设置LoDTensor的LoD。

参数：
- **lod** （List [List [int]]） - 要设置的lod。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_lod([[0, 2, 5]])

.. py:method::  set_recursive_sequence_lengths(self: paddle.fluid.core.LoDTensor, recursive_sequence_lengths: List[List[int]]) → None

根据递归序列长度recursive_sequence_lengths设置LoDTensor的LoD。

例如，如果recursive_sequence_lengths = [[2,3]]，意味着有两个长度分别为2和3的序列，相应的lod将是[[0,2,2 + 3]]，即[[0， 2,5]]。

参数：
- **recursive_sequence_lengths** （List [List [int]]） - 序列长度。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])

.. py:method::  shape(self: paddle.fluid.core_avx.Tensor) → List[int]








