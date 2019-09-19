.. _cn_api_fluid_layers_swish:

swish
-------------------------------

.. py:function:: paddle.fluid.layers.swish(x, beta=1.0, name=None)

逐元素激活 Swish 激活函数，参考 `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`_ 。

.. math::
         out = \frac{x}{1 + e^{- beta * x}}

参数：
    - **x** (Variable) -  Swish operator 的输入，数据类型为 float32，float64。
    - **beta** (float) - Swish operator 的常量beta，默认值为 1.0。
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回：
   - Variable - 数据类型以及形状和输入 **x** 一致。


**代码示例：**

.. code-block:: python
   
   # 静态图使用
   import numpy as np
   from paddle import fluid
   
   x = fluid.layers.data(name="x", shape=[3,], dtype="float32")
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
  
.. code-block:: python

  # 动态图使用
  import numpy as np
  from paddle import fluid
  import paddle.fluid.dygraph as dg
  
  data = np.random.randn(2, 3).astype("float32")
  with dg.guard(place) as g:
      x = dg.to_variable(data)
      y = fluid.layers.swish(x)
      y_np = y.numpy()
  data
  # array([[-0.0816701 ,  1.1603649 , -0.88325626],
  #        [ 0.7522361 ,  1.0978601 ,  0.12987892]], dtype=float32)
  y_np
  # array([[-0.03916847,  0.8835007 , -0.25835553],
  #        [ 0.51126915,  0.82324016,  0.06915068]], dtype=float32)
  

