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








