.. _cn_api_fluid_clip_GradientClipByNorm:

GradientClipByNorm
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByNorm(clip_norm, need_clip=None)

:alias_main: paddle.nn.GradientClipByNorm
:alias: paddle.nn.GradientClipByNorm,paddle.nn.clip.GradientClipByNorm
:old_api: paddle.fluid.clip.GradientClipByNorm



将输入的多维Tensor :math:`X` 的L2范数限制在 ``clip_norm`` 范围之内。

- 如果L2范数大于 ``clip_norm`` ，则该 Tensor 会乘以一个系数进行压缩

- 如果L2范数小于或等于 ``clip_norm`` ，则不会进行任何操作。

输入的 Tensor 不是从该类里传入， 而是默认会选择 ``Program`` 中全部的梯度，如果 ``need_clip`` 不为None，则可以只选择部分参数进行梯度裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如： :ref:`cn_api_fluid_optimizer_SGDOptimizer` )。

裁剪公式如下：

.. math::

  Out=
  \left\{
  \begin{aligned}
  &  X & & if (norm(X) \leq clip\_norm)\\
  &  \frac{clip\_norm∗X}{norm(X)} & & if (norm(X) > clip\_norm) \\
  \end{aligned}
  \right.


其中 :math:`norm（X）` 代表 :math:`X` 的L2范数

.. math::
  \\norm(X) = (\sum_{i=1}^{n}|x_i|^2)^{\frac{1}{2}}\\

参数:
 - **clip_norm** (float) - 所允许的二范数最大值。
 - **need_clip** (function, optional) - 类型: 函数。用于指定需要梯度裁剪的参数，该函数接收一个 ``Parameter`` ，返回一个 ``bool`` (True表示需要裁剪，False不需要裁剪)。默认为None，此时会裁剪网络中全部参数。

**代码示例1：静态图**
 
.. code-block:: python
            
    import paddle
    import paddle.fluid as fluid
    import numpy as np
                
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(
            main_program=main_prog, startup_program=startup_prog):
        image = fluid.data(
            name='x', shape=[-1, 2], dtype='float32')
        predict = fluid.layers.fc(input=image, size=3, act='relu') #可训练参数: fc_0.w.0, fc_0.b.0
        loss = fluid.layers.mean(predict)
        
        # 裁剪网络中全部参数：
        clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
        
        # 仅裁剪参数fc_0.w_0时：
        # 为need_clip参数传入一个函数fileter_func，fileter_func接收参数的类型为Parameter，返回类型为bool
        # def fileter_func(Parameter):
        # # 可以较为方便的通过Parameter.name判断（name可以在fluid.ParamAttr中设置，默认为fc_0.w_0、fc_0.b_0）
        #   return Parameter.name=="fc_0.w_0"
        # clip = fluid.clip.GradientClipByNorm(clip_norm=1.0, need_clip=fileter_func)

        sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.1, grad_clip=clip)
        sgd_optimizer.minimize(loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    x = np.random.uniform(-100, 100, (10, 2)).astype('float32')
    exe.run(startup_prog)
    out = exe.run(main_prog, feed={'x': x}, fetch_list=loss)


**代码示例2：动态图**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    with fluid.dygraph.guard():
        linear = fluid.dygraph.Linear(10, 10)  #可训练参数: linear_0.w.0, linear_0.b.0
        inputs = fluid.layers.uniform_random([32, 10]).astype('float32')
        out = linear(fluid.dygraph.to_variable(inputs))
        loss = fluid.layers.reduce_mean(out)
        loss.backward()

        # 裁剪网络中全部参数：
        clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)

        # 仅裁剪参数linear_0.w_0时：
        # 为need_clip参数传入一个函数fileter_func，fileter_func接收参数的类型为ParamBase，返回类型为bool
        # def fileter_func(ParamBase):
        # # 可以通过ParamBase.name判断（name可以在fluid.ParamAttr中设置，默认为linear_0.w_0、linear_0.b_0）
        #   return ParamBase.name == "linear_0.w_0"
        # # 注：linear.weight、linear.bias能分别返回dygraph.Linear层的权重与偏差，也可以此来判断
        #   return ParamBase.name == linear.weight.name
        # clip = fluid.clip.GradientClipByNorm(clip_norm=1.0, need_clip=fileter_func)

        sgd_optimizer = fluid.optimizer.SGD(
          learning_rate=0.1, parameter_list=linear.parameters(), grad_clip=clip)
        sgd_optimizer.minimize(loss)