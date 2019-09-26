.. _cn_api_fluid_layers_scale:

scale
-------------------------------

.. py:function:: paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

缩放算子。

对输入Tensor进行缩放和偏置，其公式如下：

``bias_after_scale`` 为True:

.. math::
                        Out=scale*X+bias

``bias_after_scale`` 为False:

.. math::
                        Out=scale*(X+bias)

参数:
        - **x** (Variable) - 要进行缩放的多维Tensor，数据类型可以为int8，uint8，int16，int32，int64，float32，float64。
        - **scale** (float) - 缩放的比例。
        - **bias** (float) - 缩放的的偏置。 
        - **bias_after_scale** (bool) - 判断在缩放之前或之后添加偏置。为True时，先缩放再偏置；为False时，先偏置再缩放。该参数在某些情况下，对数值稳定性很有用。
        - **act** (str，可选) - 应用于输出的激活函数，如tanh、softmax、sigmoid、relu等。
        - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

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









