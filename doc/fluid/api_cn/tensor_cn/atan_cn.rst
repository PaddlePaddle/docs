.. _cn_api_tensor_atan:

atan
-------------------------------

.. py:function:: paddle.atan(x, name=None, out=None)

arctanh 激活函数。

.. math::
        out = tanh^{-1}(x)

参数:
    - **x(Variable)** - atan的输入Tensor，数据类型为 float32 或 float64
    - **name** (str|None) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值为None。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。

返回：返回类型为Variable(Tensor|LoDTensor)， 数据类型同输入一致。

**代码示例**：

.. code-block:: python

        import numpy as np
        import paddle
        import paddle.fluid as fluid

        inputs = fluid.layers.data(name="x", shape = [3], dtype='float32')
        output = paddle.atan(inputs)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        img = np.array([-0.8183, 0.4912, -0.6444, 0.0371]).astype(np.float32)

        res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
        print(res)
        #[array([-0.6858003, 0.45658287, -0.5724284, 0.03708299], dtype=float32)]
