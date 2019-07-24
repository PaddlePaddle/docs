.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.fluid.layers.shape(input)

shape层。

获得输入变量的形状。

参数：
        - **input** （Variable）-  输入的变量

返回： (Tensor），输入变量的形状

返回类型：    Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name="input", shape=[3, 100, 100], dtype="float32")
    out = fluid.layers.shape(input)





