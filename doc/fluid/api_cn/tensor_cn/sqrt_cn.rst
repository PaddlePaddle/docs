.. _cn_api_tensor_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.sqrt(x, name=None, out=None)

:alias_main: paddle.sqrt
:alias: paddle.sqrt,paddle.tensor.sqrt,paddle.tensor.math.sqrt
:update_api: paddle.fluid.layers.sqrt



计算输入的算数平方根。

.. math::
        out=\sqrt x=x^{1/2}

.. note::
    请确保输入中的数值是非负数。

参数:

    - **x** (Variable) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。

返回：返回类型为Variable(Tensor|LoDTensor)， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

        import numpy as np
        import paddle
        import paddle.fluid as fluid

        inputs = fluid.layers.data(name="x", shape = [3], dtype='float32')
        output = paddle.sqrt(inputs)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        img = np.array([0, 9, 36]).astype(np.float32)

        res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
        print(res)
        # [array([0., 3., 6.], dtype=float32)]
