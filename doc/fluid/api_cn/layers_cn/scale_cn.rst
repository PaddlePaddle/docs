.. _cn_api_fluid_layers_scale:

scale
-------------------------------

.. py:function:: paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

:alias_main: paddle.scale
:alias: paddle.scale,paddle.tensor.scale,paddle.tensor.math.scale
:old_api: paddle.fluid.layers.scale



缩放算子。

对输入Tensor进行缩放和偏置，其公式如下：

``bias_after_scale`` 为True:

.. math::
                        Out=scale*X+bias

``bias_after_scale`` 为False:

.. math::
                        Out=scale*(X+bias)

参数:
        - **x** (Variable) - 要进行缩放的多维Tensor，数据类型可以为float32，float64，int8，int16，int32，int64，uint8。
        - **scale** (float|Variable) - 缩放的比例，是一个float类型或者一个shape为[1]，数据类型为float32的Variable类型。
        - **bias** (float) - 缩放的偏置。 
        - **bias_after_scale** (bool) - 判断在缩放之前或之后添加偏置。为True时，先缩放再偏置；为False时，先偏置再缩放。该参数在某些情况下，对数值稳定性很有用。
        - **act** (str，可选) - 应用于输出的激活函数，如tanh、softmax、sigmoid、relu等。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回: 缩放后的输出Tensor。

返回类型:  Variable(Tensor|LoDTensor)。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
     
    inputs = fluid.layers.data(name="x", shape=[2, 3], dtype='float32')
    output = fluid.layers.scale(inputs, scale = 2.0, bias = 1.0)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)

    res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
    print(res) # [array([[ 3.,  5.,  7.], [ 9., 11., 13.]], dtype=float32)]

.. code-block:: python

    # scale with parameter scale as Variable
    import paddle.fluid as fluid
    import numpy as np

    inputs = fluid.layers.data(name="x", shape=[2, 3], dtype='float32')
    scale = fluid.layers.data(name="scale", shape=[1], dtype='float32',
                              append_batch_size=False)
    output = fluid.layers.scale(inputs, scale = scale, bias = 1.0)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    scale_np = np.array([2.]).astype(np.float32)

    res = exe.run(fluid.default_main_program(), feed={'x':img, 'scale':scale_np}, fetch_list=[output])
    print(res) # [array([[ 3.,  5.,  7.], [ 9., 11., 13.]], dtype=float32)]

