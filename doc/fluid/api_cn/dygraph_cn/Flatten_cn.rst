.. _cn_api_fluid_dygraph_Flatten:

Flatten
-------------------------------

.. py:class:: paddle.nn.Flatten(start_axis=1, stop_axis=-1)



该接口用于构建 ``Flatten`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。根据给定的start_axis 和 stop_axis 将连续的维度展平。


参数
:::::::::
    - **start_axis** (int): 展开的起始维度
    - **stop_axis** (int): 展开的结束维度

形状
:::::::::
    - **x** (Tensor): 输入的Tensor


代码示例
:::::::::

.. code-block:: python
    
    import paddle
    from paddle import to_variable
    import numpy as np

    inp_np = np.ones([5, 2, 3, 4]).astype('float32')

    paddle.disable_static()

    inp_np = to_variable(inp_np)
    flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
    flatten_res = flatten(inp_np)

