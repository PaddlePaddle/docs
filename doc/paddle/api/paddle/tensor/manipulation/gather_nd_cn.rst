.. _cn_api_tensor_cn_gather_nd:

gather_nd
-------------------------------
.. py:function:: paddle.gather_nd(x, index, name=None)


该OP是 :code:`gather` 的高维推广，并且支持多轴同时索引。 :code:`index` 是一个K维度的张量，它可以认为是从 :code:`x` 中取K-1维张量，每一个元素是一个切片：

.. math::
    output[(i_0, ..., i_{K-2})] = x[index[(i_0, ..., i_{K-2})]]

显然， :code:`index.shape[-1] <= x.rank` 并且输出张量的维度是 :code:`index.shape[:-1] + x.shape[index.shape[-1]:]` 。 

示例：

::

         给定:
             x = [[[ 0,  1,  2,  3],
                       [ 4,  5,  6,  7],
                       [ 8,  9, 10, 11]],
                      [[12, 13, 14, 15],
                       [16, 17, 18, 19],
                       [20, 21, 22, 23]]]
             x.shape = (2, 3, 4)

         - 案例 1:
             index = [[1]]
             
             gather_nd(x, index)  
                      = [x[1, :, :]] 
                      = [[12, 13, 14, 15],
                         [16, 17, 18, 19],
                         [20, 21, 22, 23]]

         - 案例 2:

             index = [[0,2]]
             gather_nd(x, index)
                      = [x[0, 2, :]]
                      = [8, 9, 10, 11]

         - 案例 3:

             index = [[1, 2, 3]]
             gather_nd(x, index)
                      = [x[1, 2, 3]]
                      = [23]


参数：
    - **x** (Tensor) - 输入Tensor，数据类型可以是int32，int64，float32，float64, bool。
    - **index** (Tensor) - 输入的索引Tensor，其数据类型int32或者int64。它的维度 :code:`index.rank` 必须大于1，并且 :code:`index.shape[-1] <= x.rank` 。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：shape为index.shape[:-1] + x.shape[index.shape[-1]:]的Tensor，数据类型与 :code:`x` 一致。


**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    
    np_x = np.array([[[1, 2], [3, 4], [5, 6]],
                     [[7, 8], [9, 10], [11, 12]]])
    np_index = [[0, 1]]
    x = paddle.to_tensor(np_x)
    index = paddle.to_tensor(np_index)
    
    output = paddle.gather_nd(x, index) #[[3, 4]]


