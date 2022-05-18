.. _cn_overview_paddle_incubate_autograd:

paddle.incubate.autograd
---------------------

paddle.incubate.autograd 目录下包含飞桨框架支持的探索性的自动微分相关的API。具体如下：

-  :ref:`自动微分机制切换API <mode_switching_apis>`
-  :ref:`自动微分基础算子与原生算子转换API <transform_apis>`
-  :ref:`函数式自动微分API <functional_apis>`

.. _mode_swithing_apis:

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
特别地，在输入数量大于输出数量时，反向微分更加高效；在输入数量小于输出数量时，前向微分更加高效。在高阶微分计算中，随着阶数的升高，输出数量会越来越多，前向微分重要性也会越来越高。
为了更好地支持这些应用场景，需要深度学习框架具备高阶自动微分的能力，且支持前向和反向两种微分模式。


我们在框架中增加如下功能：

- 设计一套自动微分基础算子
- 定义框架原生算子体系和自动微分基础算子体系之间的转化规则，并实现转化逻辑
- 在自动微分基础算子上定义前向和反向自动微分规则，并实现程序变换逻辑

自动微分基础算子设计：
自动微分基础算子和原生算子共用一套标准化中间表示（ProgramDesc），但是与原生算子体系中的算子不同，这些自动微分基础算子不包含kernel实现的，只用做表达语义，用于和原生算子体系之间转化和自动微分规则变换的定义，不能直接执行。


原生算子体系和基础算子体系转化：
原生算子体系中的算子语义往往比较复杂，需要拆分为多个基础算子的组合。另一方面，自动微分基础算子由于没有kernel实现，不能直接执行，在进行完自动微分变换之后，需要转化为同语义的原生算子才可以执行。
我们通过定义原生算子和基础算子之间的转化规则，并且设计orig2prim和prim2orig两个变化分别完成原生算子到自动微分基础算子和自动微分基础算子到原生算子之间的转化。

前向和反向微分规则和程序变换逻辑：
我们在基础算子上定义linearize和transpose规则，其中单独使用linearize规则可以实现前向自动微分变换逻辑，linearize和transpose规则配合实现反向自动微分变换逻辑。规则变化具备可组合性，嵌套使用以实现高阶自动微分。linearize 和 transpose变换的想法借鉴了JAX ``https://github.com/google/jax`` 。


基于以上功能，可以实现可扩展的不限阶数的自动微分机制，支持前向微分和反向微分两种模式。


接口设计与使用案例
==========================
当前阶段我在框架中兼容性地支持两种自动微分机制，及原生的自动微分机制和基于自动微分基础算子的自动微分机制，通过全局切换接口 ``enable_prim`` 和 ``disable_prim`` 进行切换。

基于orig2prim, linearize 和 transpose三种变换改写gradients 和 minimize接口， 并且对外提供prim2orig接口。使得用户只需要做很少的改动就可以使用新自动微分机制。

.. code-block:: python

    import paddle
    from paddle.incubate.autograd import enable_prim, prim_enabled, prim2orig
    
    paddle.enable_static()
    enable_prim()
    
    x = paddle.ones(shape=[2, 2], dtype='float32')
    x.stop_gradients = False
    y = x * x
    dy_dx = paddle.static.gradients(y, x)
    if prim_enabled():
        prim2orig()
