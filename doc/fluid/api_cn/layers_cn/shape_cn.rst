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
    input = fluid.layers.data(
        name="input", shape=[3, 100, 100], dtype="float32")
    out = fluid.layers.shape(input)

