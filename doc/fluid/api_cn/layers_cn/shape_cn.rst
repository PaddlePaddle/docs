.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.fluid.layers.shape(input)

shape层。

获得输入Tensor的shape。

参数：
        - **input** （Variable）-  输入的多维Tensor，数据类型为int32，int64，float32，float64。

返回： 一个Tensor，表示输入Tensor的shape。

返回类型： Variable(Tensor)。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    inputs = fluid.layers.data(name="x", shape=[3, 100, 100], dtype="float32")
    output = fluid.layers.shape(inputs)
    
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    img = np.ones((3, 100, 100)).astype(np.float32)

    res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
    print(res) # [array([  3, 100, 100], dtype=int32)]
