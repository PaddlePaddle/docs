.. _cn_overview_paddle_incubate_autograd:

paddle.incubate.autograd
---------------------

paddle.incubate.autograd 目录下包含飞桨框架提供的自动微分相关的一些探索性API。具体如下：

-  :ref:`自动微分机制切换API <mode_switching_apis>`
-  :ref:`自动微分基础算子与原生算子转换API <transform_apis>`
-  :ref:`函数式自动微分API <functional_apis>`


.. _mode_switching_apis:

自动微分机制切换API
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.incubate.autograd.enable_prim <cn_api_paddle_incubate_autograd_enable_prim>` ", "开启基于自动微分基础算子的自动微分机制"
    " :ref:`paddle.incubate.autograd.disable_prim <cn_api_paddle_incubate_autograd_disable_prim>` ", "关闭基于自动微分基础算子的自动微分机制"
    " :ref:`paddle.incubate.autograd.prim_enabled <cn_api_paddle_incubate_autograd_prim_enabled>` ", "显示是否开启了基于自动微分基础算子的自动微分机制"


.. _transform_apis:

自动微分基础算子与原生算子转换API
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.incubate.autograd.prim2orig <cn_api_paddle_incubate_autograd_prim2orig>` ", "自动微分基础算子转换为等价功能原生算子"


.. _functional_apis:

函数式自动微分API
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.incubate.autograd.jvp <cn_api_paddle_incubate_autograd_jvp>` ", "雅可比矩阵与向量乘积"
    " :ref:`paddle.incubate.autograd.vjp <cn_api_paddle_incubate_autograd_vjp>` ", "向量与雅可比矩阵乘积"
    " :ref:`paddle.incubate.autograd.Jacobian <cn_api_paddle_incubate_autograd_Jacobian>` ", "雅可比矩阵"
    " :ref:`paddle.incubate.autograd.Hessian <cn_api_paddle_incubate_autograd_Hessian>` ", "海森矩阵"


基于自动微分基础算子的自动微分机制
==========================
在传统的深度学习任务中，神经网络的搭建分为前向和反向过程。通过深度学习框架的自动微分机制，对前向网络中的算子求一阶导数可以完成反向过程的搭建。
在一些复杂的深度学习任务中，有时会使用到高阶导数。在科学计算领域的深度学习任务中，由于引入偏微分方程组，往往需要使用到高阶导数。
特别地，在输入数量大于输出数量时，反向微分更加高效；在输入数量小于输出数量时，前向微分更加高效.
在高阶微分计算中，随着阶数的升高，输出数量会越来越多，前向微分重要性也会越来越高。
为了更好地支持这些应用场景，需要深度学习框架具备高阶自动微分的能力，且支持前向和反向两种微分模式。


在框架中增加如下功能：

- 设计一套自动微分基础算子
- 定义框架原生算子体系和自动微分基础算子体系之间的转化规则，并实现对应的程序变换
- 在自动微分基础算子上定义自动微分规则，并实现对应的程序变换

自动微分基础算子设计：
自动微分基础算子和原生算子基于同样的数据结构，但是与原生算子体系中的算子不同，这些自动微分基础算子不包含 kernel 实现，只用做表达语义，用于和原生算子体系之间转化规则和自动微分规则的定义，不能直接执行。


原生算子体系和自动微分基础算子体系之间的转化：
一方面，原生算子体系中的算子语义往往比较复杂，需要拆分为多个自动微分基础算子的组合。
另一方面，自动微分基础算子由于没有kernel实现，不能直接执行，在进行完自动微分变换之后，需要转化为同语义的原生算子才可以执行。
通过定义原生算子和自动微分基础算子之间的转化规则，在程序变换 orig2prim 和 prim2orig 中应用对应的规则，分别完成原生算子到自动微分基础算子和自动微分基础算子到原生算子之间的转化。

自动微分规则及其对应的程序变换：
在自动微分基础算子上定义 linearize 和 transpose 规则。
其中单独使用 linearize 规则可以实现前向自动微分变换，配合使用 linearize 规则和 transpose 规则可以实现反向自动微分变换。
linearize 和 transpose 程序变换的想法来自 `JAX <https://github.com/google/jax>`_ 。
规则变化具备可组合性，例如在使用 linearize 和 transpose 完成一阶反向自动微分变换之后，可以在生成的计算图上再次使用 linearize 和 transpose 规则得到二阶反向微分计算图，从而实现高阶自动微分功能。



接口设计与使用案例
==========================
当前阶段我们优先在静态图中支持了基于自动微分基础算子的自动微分机制，通过全局切换接口 ``enable_prim`` 和 ``disable_prim`` 可以在这套自动微分机制和原始的自动微分机制之间进行切换。

接口层面，我们基于 orig2prim，linearize 和 transpose 三种变换改写了 ``paddle.static.gradients`` 接口和优化器中的 ``minimize`` 接口，并且对外提供 ``prim2orig`` 接口, 只需要做很少的改动就可以使用新自动微分机制完成自动微分功能。

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.incubate.autograd import enable_prim, prim_enabled, prim2orig
    
    paddle.enable_static()
    enable_prim()
    
    x = np.random.rand(2, 20)
    
    # Set place and excutor
    place = paddle.CPUPlace()
    if paddle.device.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    
    # Build program
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        # Set input and parameter
        input_x = paddle.static.data('x', [2, 20], dtype='float64')
        input_x.stop_gradient = False
        params_w = paddle.static.create_parameter(
            shape=[20, 2], dtype='float64', is_bias=False)
        params_bias = paddle.static.create_parameter(
            shape=[2], dtype='float64', is_bias=True)
    
        # Build network
        y = paddle.tanh(paddle.matmul(input_x, params_w) + params_bias)
        dy_dx, = paddle.static.gradients([y], [input_x])
        loss = paddle.norm(dy_dx, p=2)
        opt = paddle.optimizer.Adam(0.01)
        _, grads = opt.minimize(loss)
    
        # Do prim2orig transform.
        if prim_enabled():
            prim2orig(main.block(0))
    
    # Run program
    exe.run(startup)
    grads = exe.run(main,
                    feed={'x': x},
                    fetch_list=grads)

演进计划
==========================
目前基于自动微分基础算子的自动微分机制还在积极演进阶段，可预见的工作包括：

- 提供前向微分相关API
- 适配函数式自动微分API
- 功能覆盖更多的组网API
- 支持控制流
- 支持动态图模式

欢迎持续关注或者参与共建。
