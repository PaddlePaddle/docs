.. _cn_api_tensor_linspace:

linspace
-------------------------------

.. py:function:: paddle.linspace(start, stop, num, dtype, out=None, device=None, name=None)

:alias_main: paddle.linspace
:alias: paddle.linspace,paddle.tensor.linspace,paddle.tensor.creation.linspace
:update_api: paddle.fluid.layers.linspace



该OP在给定区间内返回固定数目的均匀间隔的值。

**注意：该OP不进行梯度计算**
 
参数：
    - **start** (float|Variable) – start是区间开始的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32或者是float64。
    - **stop** (float|Variable) – end是区间结束的变量，可以是一个浮点标量，或是一个shape为[1]的Tensor，该Tensor的数据类型可以是float32或者是float64。
    - **num** (int|Variable) – num是给定区间内需要划分的区间数，可以是一个整型标量，或是一个shape为[1]的Tensor，该Tensor的数据类型需为int32。
    - **dtype** (string) – 输出Tensor的数据类型，可以是‘float32’或者是‘float64’。
    - **out** (Variable，可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **device** (str，可选) – 选择在哪个设备运行该操作，可选值包括None，'cpu'和'gpu'。如果 ``device``  为None，则将选择运行Paddle程序的设备，默认为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出结果的数据类型是float32或float64，表示等间隔划分结果的1-D Tensor，该Tensor的shape大小为 :math:`[num]` ，在mum为1的情况下，仅返回包含start元素值的Tensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    data = paddle.linspace(0, 10, 5, dtype='float32')
    data = paddle.linspace(0, 10, 1, dtype='float32')

