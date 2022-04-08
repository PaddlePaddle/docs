.. _cn_api_fluid_layers_logspace:

logspace
-------------------------------

.. py:function:: paddle.logspace(start, stop, num, base=10.0, dtype=None, name=None)

该OP返回一个Tensor，Tensor的值为在区间 :math:`[base^{start}, base^{stop}]` 上按对数均匀间隔的 :math:`num` 个值，输出Tensor的长度为num。
**注意：该OP不进行梯度计算**
 
参数
::::::::::::

    - **start** (int|float|Tensor) – ``start`` 是区间开始值以 ``base`` 为底的指数，可以是一个标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32，float64，int32 或者int64。
    - **stop** (int|float|Tensor) – ``stop`` 是区间结束值以 ``base`` 为底的指数，可以是一个标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32，float64，int32或者int64。
    - **num** (int|Tensor) – ``num`` 是给定区间内需要划分的区间数，可以是一个整型标量，或是一个shape为[1]的Tensor，该Tensor的数据类型需为int32。
    - **base** (int|float|Tensor) – ``base`` 是对数函数的底数，可以是一个标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32，float64，int32或者int64。
    - **dtype** (np.dtype|str, 可选) – 输出Tensor的数据类型，可以是float32，float64， int32或者int64。如果dtype的数据类型为None，输出Tensor数据类型为float32。
    - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::::
等对数间隔划分的1-D Tensor，该Tensor的shape大小为 :math:`[num]` ，在num为1的情况下，仅返回包含 :math:`base^{start}` 值的Tensor。


代码示例
::::::::::::

.. code-block:: python

      import paddle
      data = paddle.logspace(0, 10, 5, 2, 'float32')
      # [1.          , 5.65685415  , 32.         , 181.01933289, 1024.       ]
      data = paddle.logspace(0, 10, 1, 2, 'float32')
      # [1.]
