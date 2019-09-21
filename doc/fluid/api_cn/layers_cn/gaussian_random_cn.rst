.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
-------------------------------

.. py:function:: paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32')

生成数据符合高斯随机分布的张量。

参数：
        - **shape** （Tuple[int]）- 生成张量的形状。
        - **mean** （float）- 随机张量的均值，默认值为 0.0。
        - **std** （float）- 随机张量的标准差，默认值为 1.0。
        - **seed** （int）- 随机数种子，默认值为0。注：seed 设置为 0 表示使用系统的随机数种子。注意如果 seed 不为 0，则此算子每次将始终生成相同的随机数。
        - **dtype** （np.dtype，core.VarDesc.VarType，str）- 输出张量的数据类型，可选值为 float32，float64。

返回：        

        - Variable - 符合高斯分布的随机张量。形状为 shape，数据类型为 dtype。


**代码示例：**

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






