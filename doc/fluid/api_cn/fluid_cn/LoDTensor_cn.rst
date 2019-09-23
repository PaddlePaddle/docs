.. _cn_api_fluid_LoDTensor:

LoDTensor
-------------------------------

.. py:class:: paddle.fluid.LoDTensor


LoDTensor是一个具有LoD（Level of Details）信息的张量（Tensor），可用于表示变长序列，详见 :ref:`cn_user_guide_lod_tensor` 。

LoDTensor可以通过 ``np.array(lod_tensor)`` 方法转换为numpy.ndarray。

如果您不需要了解LoDTensor的细节，可以跳过以下的注解。

下面以两个例子说明如何用LoDTensor表示变长序列。

示例1：

假设x为一个表示变长序列的LoDTensor，它包含2个逻辑子序列，第一个序列长度是2（样本数量为2），第二个序列长度是3，总序列长度为5。
第一个序列的数据为[1, 2], [3, 4]，第二个序列的数据为[5, 6], [7, 8], [9, 10]，每个样本数据的维度均是2，该LoDTensor最终的shape为[5, 2]，其中5为总序列长度，2为每个样本数据的维度。

在逻辑上，我们可以用两种方式表示该变长序列，一种是递归序列长度的形式，即x.recursive_sequence_length = [[2, 3]]；另一种是偏移量的形式，即x.lod = [[0, 2, 2+3]]。
这两种表示方式是等价的，您可以通过LoDTensor的相应接口来设置和获取recursive_sequence_length或LoD。

在实现上，为了获得更快的序列访问速度，Paddle采用了偏移量的形式来存储不同的序列长度。因此，对recursive_sequence_length的操作最终将转换为对LoD的操作。

::

  x.data = [[1, 2], [3, 4], 
            [5, 6], [7, 8], [9, 10]]

  x.shape = [5, 2]

  x.recursive_sequence_length = [[2, 3]]

  x.lod  =  [[0, 2, 5]] 

示例2：

LoD可以有多个level（例如，一个段落可以有多个句子，一个句子可以有多个单词）。假设y为LoDTensor ，lod_level为2。从level=0来看有2个逻辑序列，序列长度分别为2和1，表示第一个逻辑序列包含2个子序列，第二个逻辑序列包含1个子序列。从level=1来看，第一个逻辑序列包含的2个子序列长度分别为2和2，第二个逻辑序列包含的1个子序列长度为3。

因此，该LoDTensor以递归序列长度形式表示为 y.recursive_sequence_length = [[2, 1], [2, 2, 3]]；相应地，以偏移量形式表示为 y.lod = [[0, 2, 3], [0, 2, 4, 7]]。

::

  y.data = [[1, 2], [3, 4], 
            [5, 6], [7, 8], 
            [9, 10], [11, 12], [13, 14]]

  y.shape = [2+2+3, 2]

  y.recursive_sequence_length = [[2, 1], [2, 2, 3]]

  y.lod = [[0, 2, 3], [0, 2, 4, 7]]

**示例代码**

.. code-block:: python

      import paddle.fluid as fluid
     
      t = fluid.LoDTensor()


.. py:method:: has_valid_recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor) → bool

该接口检查LoDTensor的LoD的正确性。

返回：   是否带有正确的LoD。

返回类型：  bool。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])
            print(t.has_valid_recursive_sequence_lengths())  # True

.. py:method::  lod(self: paddle.fluid.core_avx.LoDTensor) → List[List[int]]

该接口返回LoDTensor的LoD。

返回：LoDTensor的LoD。

返回类型：List [List [int]]。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_lod([[0, 2, 5]])
            print(t.lod()) # [[0, 2, 5]]

.. py:method:: recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor) → List[List[int]]

该接口返回与LoDTensor的LoD对应的递归序列长度。

返回：LoDTensor的LoD对应的递归序列长度。

返回类型：List [List [int]]。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])
            print(t.recursive_sequence_lengths())  # [[2, 3]]


.. py:method::  set(*args, **kwargs)
    
该接口根据输入的numpy array和设备place，设置LoDTensor的数据。

重载函数：

1. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float32], place: paddle::platform::CPUPlace) -> None

2. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int32], place: paddle::platform::CPUPlace) -> None

3. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float64], place: paddle::platform::CPUPlace) -> None

4. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int64], place: paddle::platform::CPUPlace) -> None

5. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[bool], place: paddle::platform::CPUPlace) -> None

6. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint16], place: paddle::platform::CPUPlace) -> None

7. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint8], place: paddle::platform::CPUPlace) -> None

8. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int8], place: paddle::platform::CPUPlace) -> None

9. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float32], place: paddle::platform::CUDAPlace) -> None

10. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int32], place: paddle::platform::CUDAPlace) -> None

11. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float64], place: paddle::platform::CUDAPlace) -> None

12. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int64], place: paddle::platform::CUDAPlace) -> None

13. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[bool], place: paddle::platform::CUDAPlace) -> None

14. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint16], place: paddle::platform::CUDAPlace) -> None

15. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint8], place: paddle::platform::CUDAPlace) -> None

16. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int8], place: paddle::platform::CUDAPlace) -> None

17. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float32], place: paddle::platform::CUDAPinnedPlace) -> None

18. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int32], place: paddle::platform::CUDAPinnedPlace) -> None

19. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[float64], place: paddle::platform::CUDAPinnedPlace) -> None

20. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int64], place: paddle::platform::CUDAPinnedPlace) -> None

21. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[bool], place: paddle::platform::CUDAPinnedPlace) -> None

22. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint16], place: paddle::platform::CUDAPinnedPlace) -> None

23. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[uint8], place: paddle::platform::CUDAPinnedPlace) -> None

24. set(self: paddle.fluid.core_avx.Tensor, array: numpy.ndarray[int8], place: paddle::platform::CUDAPinnedPlace) -> None

参数：
    - **array** (numpy.ndarray) - 要设置的numpy array，支持的数据类型为bool, float32, float64, int8, int32, int64, uint8, uint16。
    - **place** (CPUPlace|CUDAPlace|CUDAPinnedPlace) - 要设置的LoDTensor所在的设备。

返回：空。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())


.. py:method::  set_lod(self: paddle.fluid.core_avx.LoDTensor, lod: List[List[int]]) → None

该接口设置LoDTensor的LoD。

参数：
    - **lod** （List [List [int]]） - 要设置的LoD。

返回：空。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_lod([[0, 2, 5]])



.. py:method::  set_recursive_sequence_lengths(self: paddle.fluid.core_avx.LoDTensor, recursive_sequence_lengths: List[List[int]]) → None

该接口根据递归序列长度 ``recursive_sequence_lengths`` 设置LoDTensor的LoD。

例如，如果 ``recursive_sequence_lengths = [[2, 3]]``，意味着有两个长度分别为2和3的序列，相应的LoD是[[0, 2, 2 + 3]]，即[[0, 2, 5]]。

参数：
  - **recursive_sequence_lengths** (List [List [int]]) - 递归序列长度。

返回：空。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            t.set_recursive_sequence_lengths([[2, 3]])

.. py:method::  shape(self: paddle.fluid.core_avx.Tensor) → List[int]

该接口返回LoDTensor的shape。

返回：LoDTensor的shape。

返回类型：List[int] 。

**示例代码**

.. code-block:: python
            
            import paddle.fluid as fluid
            import numpy as np
     
            t = fluid.LoDTensor()
            t.set(np.ndarray([5, 30]), fluid.CPUPlace())
            print(t.shape())  # [5, 30]



