.. _cn_api_fluid_layers_linspace:

linspace
-------------------------------

.. py:function:: paddle.fluid.layers.linspace(start, stop, num, dtype)

该OP在给定区间内返回固定数目的均匀间隔的值。
 
参数：
    - **start** (float|Variable) – start是区间开始的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32或者是float64。
    - **stop** (float|Variable) – end是区间结束的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32或者是float64。
    - **num** (int|Variable) – num是给定区间内需要划分的区间数，可以是一个整型标量，或是一个shape为[1]的Tensor，该Tensor的数据类型需为int32。
    - **dtype** (string) – 输出Tensor的数据类型，可以是‘float32’或者是‘float64’。

返回：表示等间隔划分结果的1-D Tensor，该Tensor的shape大小为 :math:`[num]` ，在mum为1的情况下，仅返回包含start元素值的Tensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

      import paddle.fluid as fluid
      data = fluid.layers.linspace(0, 10, 5, 'float32') # [0.0,  2.5,  5.0,  7.5, 10.0]
      data = fluid.layers.linspace(0, 10, 1, 'float32') # [0.0]





