.. _cn_api_paddle_gather_nd:

gather_nd
-------------------------------
.. py:function:: paddle.gather_nd(x, index, name=None)


:code:`gather` 的高维推广，并且支持多轴同时索引。:code:`index` 是一个 K 维度的 Tensor，它可以认为是从 :code:`x` 中取 K-1 维 Tensor，每一个元素是一个切片：

.. math::
    output[(i_0, ..., i_{K-2})] = x[index[(i_0, ..., i_{K-2})]]

显然，:code:`index.shape[-1] <= x.rank` 并且输出 Tensor 的维度是 :code:`index.shape[:-1] + x.shape[index.shape[-1]:]` 。

示例：

::

         给定：
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


**示例图解说明**：

    下图展示了示例中的情形——一个形状为[2,3,4]的三维张量通过gather_nd操作分别输出三个不同的张量。通过比较，可以看出不同index下输出张量的差别。

    .. figure:: ../../images/api_legend/gather_nd.png
       :width: 761
       :alt: 示例图示
       :align: center


参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型可以是 int32、int64、float16、float32、float64、bool。
    - **index** (Tensor) - 输入的索引 Tensor，其数据类型 int32 或者 int64。它的维度 :code:`index.rank` 必须大于 1，并且 :code:`index.shape[-1] <= x.rank` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

shape 为 index.shape[:-1] + x.shape[index.shape[-1]:]的 Tensor，数据类型与 :code:`x` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.gather_nd
