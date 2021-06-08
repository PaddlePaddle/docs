.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32', name=None)




该OP返回数值符合高斯随机分布的Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数：
    - **shape** (list|tuple|Tensor) - 生成的随机Tensor的形状。如果 ``shape`` 是list、tuple，则其中的元素可以是int，或者是形状为[1]且数据类型为int32、int64的Tensor。如果 ``shape`` 是Tensor，则是数据类型为int32、int64的1-D Tensor。
    - **mean** (float|int, 可选) - 输出Tensor的均值，支持的数据类型：float、int。默认值为0.0。
    - **std** (float|int, 可选) - 输出Tensor的标准差，支持的数据类型：float、int。默认值为1.0。
    - **seed** (int, 可选) - 随机数种子，默认值为 0。注：seed 设置为 0 表示使用系统的随机数种子。注意如果 seed 不为 0，则此算子每次将始终生成相同的随机数。
    - **dtype** (str|np.dtype|core.VarDesc.VarType, 可选) - 输出Tensor的数据类型，支持float32、float64。默认值为float32。
    - **name** (str, 可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：
    Tensor：符合高斯随机分布的Tensor，形状为 ``shape``，数据类型为 ``dtype``。

抛出异常：
  - ``TypeError`` - 如果 ``shape`` 的类型不是list、tuple、Tensor。
  - ``TypeError`` - 如果 ``dtype`` 不是float32、float64。

**代码示例**：

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
	
	
.. code-block:: python

    # 动态图使用
    import numpy as np
    from paddle import fluid
    import paddle.fluid.dygraph as dg
    
    place = fluid.CPUPlace()
    with dg.guard(place) as g:
        x = fluid.layers.gaussian_random((2, 4), mean=2., dtype="float32", seed=10)
        x_np = x.numpy()       
    x_np
    # array([[2.3060477 , 2.676496  , 3.9911983 , 0.9990833 ],
    #        [2.8675377 , 2.2279181 , 0.79029655, 2.8447366 ]], dtype=float32)






