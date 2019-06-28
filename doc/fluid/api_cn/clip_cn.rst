#################
 fluid.clip
#################



.. _cn_api_fluid_clip_ErrorClipByValue:

ErrorClipByValue
-------------------------------

.. py:class:: paddle.fluid.clip.ErrorClipByValue(max, min=None)

将张量值的范围压缩到 [min, max]。


给定一个张量 ``t`` ，该操作将它的值压缩到 ``min`` 和 ``max``  之间

- 任何小于min（最小值）的值都被设置为min

- 任何大于max（最大值）的值都被设置为max

参数:
 - **max** (foat) - 要修剪的最大值。
 - **min** (float) - 要修剪的最小值。如果用户没有设置，将被框架默认设置为 ``-max`` 
  
**代码示例**
 
.. code-block:: python
        
     import paddle.fluid as fluid
     BATCH_SIZE = 128
     CLIP_MAX = 2e-6
     CLIP_MIN = -1e-6
     prog = fluid.framework.Program()
     with fluid.program_guard(main_program=prog):
        image = fluid.layers.data(name='x', shape=[784], dtype='float32')
        hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
        hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
        predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
        label = fluid.layers.data(name='y', shape=[1], dtype='int64')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
     prog_clip = prog.clone()
     prog_clip.block(0).var(hidden1.name)._set_error_clip(
        fluid.clip.ErrorClipByValue(
            max=CLIP_MAX, min=CLIP_MIN)







.. _cn_api_fluid_clip_GradientClipByGlobalNorm:

GradientClipByGlobalNorm
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm, group_name='default_group')
 
通过多个张量的范数之和的比率来剪切（clip）多个张量。

给定一个张量列表 :math:`t\_list` 和一个剪切比率 ``clip_norm`` ，返回一个被剪切的张量列表list_clipped和 :math:`t\_list` 中所有张量的全局范数(global_norm)。

剪切过程如下：

.. math::
            \\t\_list[i]=t\_list[i]∗\frac{clip\_norm}{max(global\_norm,clip\_norm)}\\
            
其中：

.. math::            
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(l2norm(t\_list[i]))^2}\\


如果 :math:`clip\_norm>global\_norm` ， :math:`t\_list` 中的张量保持不变，否则它们都会按照全局比率缩减。


参数:
 - **clip_norm** (float) - 范数最大值
 - **group_name** (str, optional) - 剪切的组名
  
**代码示例**
 
.. code-block:: python
        
    import paddle.fluid as fluid
    prog = fluid.framework.Program()
    startup_program = fluid.framework.Program()
    with fluid.program_guard(
            main_program=prog, startup_program=startup_program):
        image = fluid.layers.data(name='x', shape=[784], dtype='float32')
        label = fluid.layers.data(name='y', shape=[1], dtype='int64')
        hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
        hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
        predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
    prog_clip = prog.clone()
    avg_cost_clip = prog_clip.block(0).var(avg_cost.name)
    p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

    with fluid.program_guard(main_program=prog_clip):
        fluid.clip.set_gradient_clip(
            fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
        p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)








.. _cn_api_fluid_clip_GradientClipByNorm:

GradientClipByNorm
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByNorm(clip_norm)

将张量转换为L2范数不超过 ``clip_norm`` 的张量

该operator 限制了 输入张量 :math:`X` 的L2范数不会超过 :math:`max\_norm` 。如果 :math:`X` 的 ``L2`` 范数小于或等于 :math:`max\_norm` ,输出和 :math:`X` 一样，如果 :math:`X` 的L2范数大于 :math:`max\_norm` , :math:`X` 将被线性缩放到L2范数等于 :math:`max\_norm` ,如以下公式所示:

.. math::
            \\Out = \frac{max\_norm∗X}{norm(X)}\\

其中 :math:`norm（X）` 代表 :math:`X` 的 L2 范数


参数:
 - **clip_norm** (float) - 二范数最大值

  
**代码示例**
 
.. code-block:: python
        
    import paddle.fluid as fluid
    w_param_attrs = fluid.ParamAttr(name=None,
                                    initializer=fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=0),
                                    learning_rate=1.0,
                                    regularizer=fluid.regularizer.L1Decay(1.0),
                                    trainable=True,
                                    gradient_clip=fluid.clip.GradientClipByNorm(clip_norm=2.0))
    x = fluid.layers.data(name='x', shape=[10], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)








.. _cn_api_fluid_clip_GradientClipByValue:

GradientClipByValue
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByValue(max, min=None)

将梯度值(gradient values)的范围压缩到 [min, max]。


给定一个张量 ``t`` ，该操作将它的值压缩到 ``min`` 和 ``max`` 之间

- 任何小于最小值的值都被设置为最小值

- 任何大于max的值都被设置为max

参数:
 - **max** (foat) - 要修剪的最大值。
 - **min** (float，optional) - 要修剪的最小值。如果用户没有设置，将被 ``framework`` 设置为 ``-max`` 。
  
**代码示例**
 
.. code-block:: python
        
     import paddle.fluid as fluid
     w_param_attrs = fluid.ParamAttr(name=None,
                                     initializer=fluid.initializer.UniformInitializer(low=-1.0, high=1.0, seed=0),
                                     learning_rate=1.0,
                                     regularizer=fluid.regualrizer.L1Decay(1.0),
                                     trainable=True,
                                     gradient_clip=fluid.clip.GradientClipByValue(-1.0, 1.0))
     x = fluid.layers.data(name='x', shape=[10], dtype='float32')
     y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)
     






