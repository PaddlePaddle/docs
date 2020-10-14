.. _cn_api_fluid_layers_linspace:

linspace
-------------------------------

.. py:function:: paddle.fluid.layers.linspace(start, stop, num, dtype=None, name=None)

该OP返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num。
**注意：该OP不进行梯度计算**
 
参数：
    - **start** (int|float|Tensor) – ``start`` 是区间开始的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32，float64，int32 或者int64。
    - **stop** (int|float|Tensor) – ``stop`` 是区间结束的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32，float64，int32或者int64。
    - **num** (int|Tensor) – ``num`` 是给定区间内需要划分的区间数，可以是一个整型标量，或是一个shape为[1]的Tensor，该Tensor的数据类型需为int32。
    - **dtype** (np.dtype|str, 可选) – 输出Tensor的数据类型，可以是float32，float64， int32或者int64。如果dtype的数据类型为None，输出Tensor数据类型为float32。
    - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：表示等间隔划分结果的1-D Tensor，该Tensor的shape大小为 :math:`[num]` ，在mum为1的情况下，仅返回包含start元素值的Tensor。


**代码示例**：

.. code-block:: python

      import paddle.fluid as fluid
      data = fluid.layers.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
      data = fluid.layers.linspace(0, 10, 1, 'float32') # [0.0]





