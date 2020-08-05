.. _cn_api_tensor_linspace:

linspace
-------------------------------

.. py:function:: paddle.linspace(start, stop, num, dtype=None, name=None)

:alias_main: paddle.linspace
:alias: paddle.tensor.linspace, paddle.tensor.creation.linspace



该OP返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num。

**注意：该OP不进行梯度计算**
 
参数：
    - **start** (float|Tensor) – ``start`` 是区间开始的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32或者是float64。
    - **stop** (float|Tensor) – ``end`` 是区间结束的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32或者是float64。
    - **num** (int|Tensor) – ``num`` 是给定区间内需要划分的区间数，可以是一个整型标量，或是一个shape为[1]的Tensor，该Tensor的数据类型需为int32。
    - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) – 输出Tensor的数据类型，可以是float32或者是float64。如果dtype为None，默认类型为float32。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出结果的数据类型是float32或float64，表示等间隔划分结果的1-D Tensor，该Tensor的shape大小为 :math:`[num]` ，在mum为1的情况下，仅返回包含start元素值的Tensor。

返回类型：Variable

抛出异常：
    - ``TypeError`` - 当start或者stop的数据类型不是float32或者float64。
    - ``TypeError`` - 当num的数据类型不是float32或者float64。
    - ``TypeError`` - 当dtype的类型不是float32或者float64。

**代码示例**：

.. code-block:: python

      import paddle
      data = paddle.linspace(0, 10, 5, dtype='float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
      data = paddle.linspace(0, 10, 1, dtype='float32') # [0.0]

