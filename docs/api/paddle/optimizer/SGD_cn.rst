.. _cn_api_paddle_optimizer_SGD:

SGD
-------------------------------

.. py:class:: paddle.optimizer.SGD(learning_rate=0.001, parameters=None, weight_decay=None, grad_clip=None, multi_precision=False, name=None, *, maximize=False)

随机梯度下降算法的优化器。

.. math::
            \\param\_out=param-learning\_rate*grad\\


为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters（传入 SGD 的变量，一般会是网络中的参数） ，最小化网络损失值 loss。

参数
::::::::::::

    - **learning_rate** (float|LRScheduler，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个 LRScheduler 类，默认值为 0.001。
    - **parameters** (list|tuple|None，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **weight_decay** (int|float|WeightDecayRegularizer|None，可选) - 权重衰减系数，可以是 L2 正则化系数的 int 或 float，也可以是 :ref:`cn_api_paddle_regularizer_L1Decay` 或 :ref:`cn_api_paddle_regularizer_L2Decay`。若参数已通过 :ref:`cn_api_paddle_ParamAttr` 设置正则化器，则忽略优化器中的设置；否则该设置生效。默认值为 None，即不进行正则化。
    - **grad_clip** (GradientClipBase，可选) - 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **multi_precision** (bool，可选) - 是否在参数更新时使用多精度。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::

    - **maximize** (bool，可选) - 是否对参数最大化目标函数，而非最小化。默认值为 False。



代码示例
::::::::::::

COPY-FROM: paddle.optimizer.SGD

方法
::::::::::::
step(closure=None)
''''''''''''''''''''''''''''''''''''''''

.. note::

    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

    执行一次优化器并进行参数更新。

**参数**

    - **closure** (Callable[[], Tensor], 可选) - 用于评估模型并返回损失的闭包函数。闭包函数应接受 0 个参数并返回 Tensor。适用于需要多次评估损失的优化过程。默认值为 None。

**返回**

Tensor 或 None。若传入 closure 参数则返回其输出的损失，否则返回 None。

**代码示例**

COPY-FROM: paddle.optimizer.SGD.step

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters（传入 SGD 的变量，一般会是网络中的参数），最小化网络损失值 loss。

**参数**

    - **loss** (Tensor) - 需要最小化的损失值变量。
    - **startup_program** (Program，可选) - 用于初始化 parameters 中参数的 :ref:`cn_api_paddle_static_Program`，默认值为 None，此时将使用 :ref:`cn_api_paddle_static_default_startup_program`。
    - **parameters** (list，可选) - 待更新的 Parameter 或者 Parameter.name 组成的列表，默认值为 None，此时将更新所有的 Parameter。
    - **no_grad_set** (set，可选) - 不需要更新的 Parameter 或者 Parameter.name 组成的集合，默认值为 None。

**返回**

 tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由(param, param_grad)组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

COPY-FROM: paddle.optimizer.SGD.minimize

clear_grad(set_to_zero=True)
''''''''''''''''''''''''''''''''''''''''

.. note::

    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


    清除需要优化的参数的梯度。

**参数**

    - **set_to_zero** (bool，可选) - 是否将梯度置零。若为 False，则删除梯度。默认值为 True。

**代码示例**

COPY-FROM: paddle.optimizer.SGD.clear_grad

set_lr(value)
'''''''''

.. note::

    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

set_lr_scheduler(scheduler)
'''''''''

.. note::

    该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。
