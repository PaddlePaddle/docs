.. _cn_api_tensor_argmax:

pow
-------------------------------

.. py:function:: paddle.pow(input, exponent, out=None, name=None):

该OP是指数激活算子：

.. math::
        out = x^{exponent}

参数：
    - **x** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor`` ，数据类型为 ``float32`` 或 ``float64`` 。
    - **exponent** （float32|Variable，可选）- ``float32`` 或形状为[1]的 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``。Pow OP的指数因子。默认值：1.0。
    - **out** (Variable, 可选) - 默认值None，如果out不为空，则该运算结果存储在out变量中。 
    - **name** (str，可选) - 默认值None，输出的名称。该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_name`。当out和name同时不为空时，结果输出变量名与out保持一致。

返回：维度与输入 `x` 相同的 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``x`` 相同。

返回类型：Variable。


**代码示例：**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    x = fluid.data(name="x", shape=[32,32], dtype="float32")
    
    # 示例1: 参数exponent是个浮点数
    res = fluid.data(name="output", shape=[32,32], dtype="float32")
    y_1 = paddle.pow(x, 2.0, out=res)
    # y_1 is x^{2.0}
    
    # 示例2: 参数exponent是个变量
    exponent_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
    res = fluid.data(name="output", shape=[32,32], dtype="float32")
    y_2 = paddle.pow(x, exponent_tensor, out=res)
    # y_2 is x^{3.0}