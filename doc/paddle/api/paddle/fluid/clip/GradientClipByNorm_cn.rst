.. _cn_api_fluid_clip_GradientClipByNorm:

GradientClipByNorm
-------------------------------

.. py:class:: paddle.nn.GradientClipByNorm(clip_norm, need_clip=None)




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

**代码示例1：动态图**
 
.. code-block:: python

    import paddle
    import numpy as np

    # used in default dygraph mode

    paddle.disable_static()

    in_np = np.random.uniform(-1, 1, [10, 10]).astype("float32")
    x = paddle.to_tensor(in_np)
    linear = paddle.nn.Linear(10, 10)
    out = linear(x)
    loss = paddle.mean(out)
    loss.backward()

    # 裁剪网络中全部参数：
    clip = paddle.nn.GradientClipByNorm(clip_norm=1.0)

    # 仅裁剪参数linear_0.w_0时：
    # pass a function(fileter_func) to need_clip, and fileter_func receive a ParamBase, and return bool
    # def fileter_func(ParamBase):
    # # 可以通过ParamBase.name判断（name可以在paddle.ParamAttr中设置，默认为linear_0.w_0、linear_0.b_0）
    #   return ParamBase.name == "linear_0.w_0"
    # # 注：linear.weight、linear.bias能分别返回dygraph.Linear层的权重与偏差，可以此来判断
    #   return ParamBase.name == linear.weight.name
    # clip = paddle.nn.GradientClipByNorm(clip_norm=1.0, need_clip=fileter_func)

    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
    sdg.minimize(loss)
            

**代码示例2：静态图**

.. code-block:: python

    import paddle
    import numpy as np

    # used in static mode

    paddle.enable_static()
                        
    main_prog = paddle.static.Program()
    start_prog = paddle.static.Program()
    with paddle.static.program_guard(main_prog, start_prog):
        x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        out = paddle.static.nn.fc(x, 100)
        loss = paddle.mean(out)
        
        # 裁剪网络中全部参数：
        clip = paddle.nn.GradientClipByNorm(clip_norm=1.0)
        
        # 仅裁剪参数fc_0.w_0时：
        # 为need_clip参数传入一个函数fileter_func，fileter_func接收参数的类型为Parameter，返回类型为bool
        # def fileter_func(Parameter):
        # # 可以较为方便的通过Parameter.name判断（name可以在paddle.ParamAttr中设置，默认为fc_0.w_0、fc_0.b_0）
        #   return Parameter.name=="fc_0.w_0"
        # clip = paddle.nn.GradientClipByNorm(clip_norm=1.0, need_clip=fileter_func)

        sdg = paddle.optimizer.SGD(learning_rate=0.1, grad_clip=clip)
        sdg.minimize(loss)

    exe = paddle.static.Executor(place=paddle.CPUPlace())
    exe.run(start_prog)
    out = exe.run(main_prog, 
                  feed={'x': np.random.uniform(-100, 100, (10, 2)).astype('float32')}, 
                  fetch_list=loss)


