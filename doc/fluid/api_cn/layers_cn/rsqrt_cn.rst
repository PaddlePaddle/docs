.. _cn_api_fluid_layers_rsqrt:

rsqrt
-------------------------------

.. py:function:: paddle.fluid.layers.rsqrt(x, name=None)

该OP为rsqrt激活函数。

注：输入x应确保为非 **0** 值，否则程序会抛异常退出。

其运算公式如下：

.. math::
    out = \frac{1}{\sqrt{x}}


参数:
    - **x** (Variable) – 输入是多维Tensor或LoDTensor，数据类型可以是float32和float64。 
    - **name** (str，可选）— 这一层的名称（可选）。如果设置为None，则将自动命名这一层。默认值为None。

返回：对输入x进行rsqrt激活函数计算后的Tensor或LoDTensor，数据shape和输入x的shape一致。

返回类型：Variable，数据类型和输入数据类型一致。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.rsqrt(data)

