.. _cn_api_fluid_layers_gather:

gather
-------------------------------

.. py:function:: paddle.fluid.layers.gather(input, index, overwrite=True)




根据索引 ``index`` 获取输入 ``input`` 的最外层维度的条目，并将它们拼接在一起。

.. math::

        Out=X[Index]

.. code-block:: text

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        Then:

        Out = [[3, 4],
               [5, 6]]


参数
::::::::::::

        - **input** (Tensor) - 输入，秩 ``rank >= 1``，支持的数据类型包括 int32、int64、float32、float64 和 uint8 (CPU)、float16（GPU） 。
        - **index** (Tensor) - 索引，秩 ``rank = 1``，数据类型为 int32 或 int64。
        - **overwrite** (bool) - 具有相同索引时在反向更新梯度的模式。如果为 ``True``，则使用覆盖模式更新相同索引的梯度；如果为 ``False``，则使用累积模式更新相同索引的梯度。默认值为 ``True`` 。

返回
::::::::::::
和输入的秩相同的输出 Tensor。


代码示例
::::::::::::

..  code-block:: python

  import paddle.fluid as fluid
  x = fluid.layers.data(name='x', shape=[-1, 5], dtype='float32')
  index = fluid.layers.data(name='index', shape=[-1, 1], dtype='int32')
  output = fluid.layers.gather(x, index)
