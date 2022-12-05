.. _cn_api_fluid_layers_swish:

swish
-------------------------------

.. py:function:: paddle.fluid.layers.swish(x, beta=1.0, name=None)




逐元素计算 Swish 激活函数，参考 `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`_ 。

.. math::
         out = \frac{x}{1 + e^{- beta * x}}

参数
::::::::::::

    - **x** (Variable) -  多维 Tensor，数据类型为 float32，float64。
    - **beta** (float) - Swish operator 的常量 beta，默认值为 1.0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    - Swish op 的结果，多维 Tensor。数据类型为 float32 或 float64，数据类型以及形状和输入 x 一致。

返回类型
::::::::::::

    - Variable


代码示例
::::::::::::

.. code-block:: python

    # 静态图使用
    import numpy as np
    from paddle import fluid

    x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
    y = fluid.layers.swish(x, beta=2.0)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    start = fluid.default_startup_program()
    main = fluid.default_main_program()

    data = np.random.randn(2, 3).astype("float32")
    exe.run(start)
    y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])

    data
    # array([[-1.1239197 ,  1.3391294 ,  0.03921051],
    #        [ 1.1970421 ,  0.02440812,  1.2055548 ]], dtype=float32)
    y_np
    # array([[-0.2756806 ,  1.0610548 ,  0.01998957],
    #        [ 0.9193261 ,  0.01235299,  0.9276883 ]], dtype=float32)

COPY-FROM: paddle.fluid.layers.swish
