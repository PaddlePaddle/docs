.. _cn_api_fluid_layers_scale:

scale
-------------------------------

.. py:function:: paddle.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)




缩放算子。

对输入Tensor进行缩放和偏置，其公式如下：

``bias_after_scale`` 为True:

.. math::
                        Out=scale*X+bias

``bias_after_scale`` 为False:

.. math::
                        Out=scale*(X+bias)

参数:
        - **x** (Tensor) - 要进行缩放的多维Tensor，数据类型可以为float32，float64，int8，int16，int32，int64，uint8。
        - **scale** (float|Tensor) - 缩放的比例，是一个float类型或者一个shape为[1]，数据类型为float32的Tensor类型。
        - **bias** (float) - 缩放的偏置。 
        - **bias_after_scale** (bool) - 判断在缩放之前或之后添加偏置。为True时，先缩放再偏置；为False时，先偏置再缩放。该参数在某些情况下，对数值稳定性很有用。
        - **act** (str，可选) - 应用于输出的激活函数，如tanh、softmax、sigmoid、relu等。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回: Tensor，缩放后的计算结果。

**代码示例：**

.. code-block:: python

    # scale as a float32 number
    import paddle

    data = paddle.randn(shape=[2,3], dtype='float32')
    res = paddle.scale(data, scale=2.0, bias=1.0)

.. code-block:: python

    # scale with parameter scale as a Tensor
    import paddle

    data = paddle.randn(shape=[2, 3], dtype='float32')
    factor = paddle.to_tensor([2], dtype='float32')
    res = paddle.scale(data, scale=factor, bias=1.0)
