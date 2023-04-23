.. _cn_api_fluid_layers_linear_lr_warmup:

linear_lr_warmup
-------------------------------

.. py:function:: paddle.fluid.layers.linear_lr_warmup(learning_rate, warmup_steps, start_lr, end_lr)





该OP使用学习率优化策略-线性学习率热身(warm up)对学习率进行初步调整。在正常调整学习率之前，先逐步增大学习率，具体原理可参考：`Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_ 

当训练步数（global_step）小于热身步数（warmup_steps）时，学习率lr按如下方式更新：

.. code-block:: text

        linear_step = end_lr - start_lr
        lr = start_lr + linear_step * (global_step / warmup_steps)

其中start_lr为warm up起始学习率，end_lr为最终学习率；

当训练步数（global_step）大于等于热身步数（warmup_steps）时，学习率lr为：

.. code-block:: text

        lr = learning_rate

其中learning_rate为热身之后的学习率。

参数
::::::::::::

    - **learning_rate** （Variable|float） - 热身之后的学习率，它可以是数据类型为float32的1D-Tensor或单个浮点数。
    - **warmup_steps** （int） - 进行warm up过程的步数。
    - **start_lr** （float） - warm up的起始学习率。
    - **end_lr** （float） - warm up的最终学习率。

返回
::::::::::::
进行热身衰减后的学习率，数据类型与learning_rate相同。

返回类型
::::::::::::
Variable


代码示例
::::::::::::

.. code-block:: python

        import paddle.fluid as fluid

        boundaries = [100, 200]
        lr_steps = [0.1, 0.01, 0.001]
        learning_rate = fluid.layers.piecewise_decay(boundaries, lr_steps) #case1, Tensor
        #learning_rate = 0.1  #case2, float32
        warmup_steps = 50
        start_lr = 1. / 3.
        end_lr = 0.1
        decayed_lr = fluid.layers.linear_lr_warmup(learning_rate,
            warmup_steps, start_lr, end_lr)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        out, = exe.run(fetch_list=[decayed_lr.name])
        print(out)
        # case1: [0.33333334]
        # case2: [0.33333334]
