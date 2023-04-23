.. _cn_api_fluid_layers_gelu:

gelu
-------------------------------

.. py:function:: paddle.fluid.layers.gelu(x)




逐元素计算 Gelu 激活函数。更多细节请参考 `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ 。

如果使用近似计算：

.. math::
    out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

如果不使用近似计算：

.. math::
    out = 0.5 * x * (1 + erf(\frac{x}{\sqrt{2}}))

参数
::::::::::::

  - **x** (Variable) - Gelu Op 的输入，多维 Tensor，数据类型为 float32 或 float64。
  - **approximate** (bool，可选) - 是否使用近似计算，默认值为 False。

返回
::::::::::::

  - 多维 Tensor，数据类型为 float32 或 float64，和输入 x 的数据类型相同，形状和输入 x 相同。

返回类型
::::::::::::

  - Variable

代码示例
::::::::::::

.. code-block:: python

    # declarative mode
    import numpy as np
    from paddle import fluid

    x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
    y = fluid.layers.gelu(x)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    start = fluid.default_startup_program()
    main = fluid.default_main_program()

    data = np.random.randn(2, 3).astype("float32")
    exe.run(start)

    y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])

    data
    # array([[ 0.87165993, -1.0541513 , -0.37214822],
    #         [ 0.15647964,  0.32496083,  0.33045998]], dtype=float32)
    y_np
    # array([[ 0.70456535, -0.15380788, -0.13207214],
    #        [ 0.08796856,  0.20387867,  0.2080159 ]], dtype=float32)

COPY-FROM: paddle.fluid.layers.gelu
