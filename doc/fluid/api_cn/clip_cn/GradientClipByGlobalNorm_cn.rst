.. _cn_api_fluid_clip_GradientClipByGlobalNorm:

GradientClipByGlobalNorm
-------------------------------

.. py:class:: paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm, group_name='default_group')
 
通过多个 Tensor 的范数之和的比率，来剪切（clip）多个 Tensor （ Tensor 不是从该类传入， 通过 ``fluid.program_guard`` 的 ``main_program`` 参数传入，即公式中的 :math:`t\_list` 见代码实例）。

给定一个 Tensor 列表 :math:`t\_list` 和一个剪切比率 ``clip_norm`` ，返回该类的实例作为 ``set_gradient_clip`` 方法的第一个参数， ``set_gradient_clip`` 第二个参数是用来计算被剪切的 Tensor 列表（该值默认为 ``None`` 会基于所有 Tensor 列表来计算全局范数 ``global_norm`` 。

剪切过程如下：

.. math::
            \\t\_list[i]=t\_list[i]∗\frac{clip\_norm}{max(global\_norm,clip\_norm)}\\
            
其中：

.. math::            
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(l2norm(t\_list[i]))^2}\\


参数:
 - **clip_norm** (float) - 范数最大值
 - **group_name** (str, optional) - 剪切的组名
  
**代码示例**
 
.. code-block:: python
        
    import paddle.fluid as fluid
    import paddle.fluid.core as core
    import paddle

    place = core.CPUPlace()
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

    p_g = fluid.backward.append_backward(loss=avg_cost)
    p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

    with fluid.program_guard(main_program=prog_clip, startup_program=startup_program):
        fluid.clip.set_gradient_clip(
            fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
        p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)

    grad_list = [elem[1] for elem in p_g]
    grad_clip_list = [elem[1] for elem in p_g_clip]

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=8192),
        batch_size=128)

    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
    exe.run(startup_program)

    count = 0
    for data in train_reader():
        count += 1
        print("count:%s" % count)
        if count > 5:
            break
        out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
        out_clip = exe.run(prog_clip,
                           feed=feeder.feed(data),
                           fetch_list=grad_clip_list)







