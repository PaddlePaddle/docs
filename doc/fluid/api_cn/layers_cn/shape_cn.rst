.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.fluid.layers.shape(input)

:alias_main: paddle.shape
:alias: paddle.shape,paddle.tensor.shape,paddle.tensor.attribute.shape
:old_api: paddle.fluid.layers.shape



shape层。

获得输入Tensor的shape。

参数：
        - **input** （Variable）-  输入的多维Tensor，数据类型为float32，float64，int32，int64。

返回： 一个Tensor，表示输入Tensor的shape。

返回类型： Variable(Tensor)。

**代码示例：**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    inputs = fluid.layers.data(name='x', shape=[3, 100, 100], dtype='float32')
    output = paddle.shape(inputs)
    
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(paddle.default_startup_program())
    img = np.ones((3, 100, 100)).astype(np.float32)
    
    res = exe.run(paddle.default_main_program(), feed={'x': img}, fetch_list=[
        output])
    print(res)

