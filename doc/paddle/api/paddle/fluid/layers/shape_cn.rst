.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.fluid.layers.shape(input)

:alias_main: paddle.shape
:alias: paddle.shape,paddle.tensor.shape,paddle.tensor.attribute.shape
:old_api: paddle.fluid.layers.shape



shape层。

获得输入Tensor或SelectedRows的shape。

::

    示例1:
        输入是 N-D Tensor类型:
            input = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        输出shape:
            input.shape = [2, 4]

    示例2:
        输入是 SelectedRows类型:
            input.rows = [0, 4, 19]
            input.height = 20
            input.value = [ [1, 2], [3, 4], [5, 6] ]  # inner tensor
        输出shape:
            input.shape = [3, 2]

参数：
        - **input** （Variable）-  输入的多维Tensor或SelectedRows，数据类型为float16，float32，float64，int32，int64。如果输入是SelectedRows类型，则返回其内部持有Tensor的shape。


返回： 一个Tensor，表示输入Tensor或SelectedRows的shape。

返回类型： Variable(Tensor)。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    inputs = fluid.data(name="x", shape=[3, 100, 100], dtype="float32")
    output = fluid.layers.shape(inputs)
    
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    img = np.ones((3, 100, 100)).astype(np.float32)

    res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
    print(res) # [array([  3, 100, 100], dtype=int32)]
