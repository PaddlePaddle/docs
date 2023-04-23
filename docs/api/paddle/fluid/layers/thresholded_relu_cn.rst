.. _cn_api_fluid_layers_thresholded_relu:

thresholded_relu
-------------------------------

.. py:function:: paddle.fluid.layers.thresholded_relu(x,threshold=None)




逐元素计算 ThresholdedRelu 激活函数。

.. math::

  out = \left\{\begin{matrix}
      x, &if x > threshold\\
      0, &otherwise
      \end{matrix}\right.

参数
::::::::::::

  - **x** (Variable) -ThresholdedRelu Op 的输入，多维 Tensor，数据类型为 float32，float64。
  - **threshold** (float，可选)-激活函数的 threshold 值，如 threshold 值为 None，则其值为 1.0。

返回
::::::::::::

   - 多维 Tensor，数据类型为 float32 或 float64，和输入 x 的数据类型相同，形状和输入 x 相同。

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
    y = fluid.layers.thresholded_relu(x, threshold=0.1)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    start = fluid.default_startup_program()
    main = fluid.default_main_program()
    data = np.random.randn(2, 3).astype("float32")
    exe.run(start)
    y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])

    data
    # array([[ 0.21134382, -1.1805999 ,  0.32876605],
    #        [-1.2210793 , -0.7365624 ,  1.0013918 ]], dtype=float32)
    y_np
    # array([[ 0.21134382, -0.        ,  0.32876605],
    #        [-0.        , -0.        ,  1.0013918 ]], dtype=float32)


COPY-FROM: paddle.fluid.layers.thresholded_relu
