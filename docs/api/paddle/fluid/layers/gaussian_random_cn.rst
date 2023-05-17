.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32', name=None)




该 OP 返回数值符合高斯随机分布的 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::::

    - **shape** (list|tuple|Tensor) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor。
    - **mean** (float|int，可选) - 输出 Tensor 的均值，支持的数据类型：float、int。默认值为 0.0。
    - **std** (float|int，可选) - 输出 Tensor 的标准差，支持的数据类型：float、int。默认值为 1.0。
    - **seed** (int，可选) - 随机数种子，默认值为 0。注：seed 设置为 0 表示使用系统的随机数种子。注意如果 seed 不为 0，则此算子每次将始终生成相同的随机数。
    - **dtype** (str|np.dtype|core.VarDesc.VarType，可选) - 输出 Tensor 的数据类型，支持 float32、float64。默认值为 float32。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    Tensor：符合高斯随机分布的 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

抛出异常
::::::::::::

  - ``TypeError`` - 如果 ``shape`` 的类型不是 list、tuple、Tensor。
  - ``TypeError`` - 如果 ``dtype`` 不是 float32、float64。

代码示例
::::::::::::

.. code-block:: python

    # 静态图使用
    import numpy as np
    from paddle import fluid

    x = fluid.layers.gaussian_random((2, 3), std=2., seed=10)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    start = fluid.default_startup_program()
    main = fluid.default_main_program()

    exe.run(start)
    x_np, = exe.run(main, feed={}, fetch_list=[x])

    x_np
    # array([[2.3060477, 2.676496 , 3.9911983],
    #        [0.9990833, 2.8675377, 2.2279181]], dtype=float32)


COPY-FROM: paddle.fluid.layers.gaussian_random
