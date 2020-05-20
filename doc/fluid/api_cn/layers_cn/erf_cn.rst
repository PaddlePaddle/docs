.. _cn_api_fluid_layers_erf:

erf
-------------------------------

.. py:function:: paddle.fluid.layers.erf(x)

:alias_main: paddle.erf
:alias: paddle.erf,paddle.tensor.erf,paddle.tensor.math.erf,paddle.nn.functional.erf,paddle.nn.functional.activation.erf
:old_api: paddle.fluid.layers.erf



逐元素计算 Erf 激活函数。更多细节请参考 `Error function <https://en.wikipedia.org/wiki/Error_function>`_ 。


.. math::
    out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

参数：
  - **x** (Variable) - Erf Op 的输入，多维 Tensor 或 LoDTensor，数据类型为 float16, float32 或 float64。

返回：
  - 多维 Tensor 或 LoDTensor, 数据类型为 float16, float32 或 float64， 和输入 x 的数据类型相同，形状和输入 x 相同。

返回类型：
  - Variable

**代码示例**：

.. code-block:: python

    # declarative mode
    import paddle
    import numpy as np
    from paddle import fluid
    
    x = paddle.data(name='x', shape=(-1, 3), dtype='float32')
    y = paddle.erf(x)
    
    place = paddle.CPUPlace()
    exe = paddle.Executor(place)
    start = paddle.default_startup_program()
    main = paddle.default_main_program()
    
    data = np.random.randn(2, 3).astype('float32')
    # array([[ 0.4643714 , -1.1509596 ,  1.2538221 ],
    #        [ 0.34369683,  0.27478245,  1.1805398 ]], dtype=float32)
    exe.run(start)
    
    y_np, = exe.run(main, feed={'x': data}, fetch_list=[y])
    
    data
    # array([[ 0.4643714 , -1.1509596 ,  1.2538221 ],
    #        [ 0.34369683,  0.27478245,  1.1805398 ]], dtype=float32)
    y_np
    # array([[ 0.48863927, -0.8964121 ,  0.9237998 ],
    #        [ 0.37307587,  0.30242872,  0.9049887 ]], dtype=float32)

.. code-block:: python

    # declarative mode
    import paddle
    import numpy as np
    from paddle import fluid
    
    x = paddle.data(name='x', shape=(-1, 3), dtype='float32')
    y = paddle.erf(x)
    
    place = paddle.CPUPlace()
    exe = paddle.Executor(place)
    start = paddle.default_startup_program()
    main = paddle.default_main_program()
    
    data = np.random.randn(2, 3).astype('float32')
    # array([[ 0.4643714 , -1.1509596 ,  1.2538221 ],
    #        [ 0.34369683,  0.27478245,  1.1805398 ]], dtype=float32)
    exe.run(start)
    
    y_np, = exe.run(main, feed={'x': data}, fetch_list=[y])
    
    data
    # array([[ 0.4643714 , -1.1509596 ,  1.2538221 ],
    #        [ 0.34369683,  0.27478245,  1.1805398 ]], dtype=float32)
    y_np
    # array([[ 0.48863927, -0.8964121 ,  0.9237998 ],
    #        [ 0.37307587,  0.30242872,  0.9049887 ]], dtype=float32)

