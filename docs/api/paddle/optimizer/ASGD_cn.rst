.. _cn_api_paddle_optimizer_ASGD:

ASGD
-------------------------------

.. py:class:: paddle.optimizer.ASGD(learning_rate=0.001, batch_num=1, parameters=None, weight_decay=None, grad_clip=None, multi_precision=False, name=None)

ASGD 算法的优化器。有关详细信息，请参阅：

`Minimizing Finite Sums with the Stochastic Average Gradient <https://hal.science/hal-00860051v2>`_ 。


.. math::

    \begin{aligned}
        &\hspace{0mm} d=0,\ y_i=0\ \textbf{for}\ i=1,2,...,n                            \\
        &\hspace{0mm} \textbf{for}\  \: m=0,1,...\ \textbf{do} \:                       \\
        &\hspace{5mm} i=m\ \%\ n                                                        \\
        &\hspace{5mm} d=d-y_i+f_i{}'(x)                                                 \\
        &\hspace{5mm} y_i=f_i{}'(x)                                                     \\
        &\hspace{5mm} x=x-learning\_rate(\frac{d}{\mathrm{min}(m+1,\ n)}+\lambda x)     \\
        &\hspace{0mm} \textbf{end for}                                                  \\
    \end{aligned}


参数
::::::::::::

    - **learning_rate** (float|Tensor|LRScheduler，可选) - 学习率，用于参数更新的计算。可以是一个浮点型值、浮点类型的 Tensor 或 LRScheduler。默认值为 0.001。
    - **batch_num** (int，可选) - 完成一个 epoch 所需迭代的次数。默认值为 1。
    - **parameters** (list|tuple|None，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **weight_decay** (int|float|WeightDecayRegularizer|None，可选) - 权重衰减策略。可以是 int 或 float 类型的 L2 正则化系数，或 :ref:`cn_api_paddle_regularizer_L1Decay`、:ref:`cn_api_paddle_regularizer_L2Decay`。若参数已通过 :ref:`cn_api_paddle_ParamAttr` 设置 regularizer，则该参数的设置会被忽略；否则在优化器中生效。默认值为 None，表示不进行正则化。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **multi_precision** (bool，可选) – 在基于 GPU 设备的混合精度训练场景中，该参数主要用于保证梯度更新的数值稳定性。设置为 True 时，优化器会针对 FP16 类型参数保存一份与其值相等的 FP32 类型参数备份。梯度更新时，首先将梯度类型提升到 FP32，然后将其更新到 FP32 类型参数备份中。最后，更新后的 FP32 类型值会先转换为 FP16 类型，再赋值给实际参与计算的 FP16 类型参数。默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


代码示例
::::::::::::

COPY-FROM: paddle.optimizer.ASGD


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

COPY-FROM: paddle.optimizer.ASGD.step

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters 中的 Parameters，最小化网络损失值 loss。

**参数**

    - **loss** (Tensor) - 需要最小化的损失值变量
    - **startup_program** (Program，可选) - 用于初始化 parameters 中参数的 :ref:`cn_api_paddle_static_Program`，默认值为 None，此时将使用 :ref:`cn_api_paddle_static_default_startup_program` 。
    - **parameters** (list，可选) - 待更新的 Parameter 或者 Parameter.name 组成的列表，默认值为 None，此时将更新所有的 Parameter。
    - **no_grad_set** (set，可选) - 不需要更新的 Parameter 或者 Parameter.name 组成的集合，默认值为 None。

**返回**

 tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由(param, param_grad)组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

COPY-FROM: paddle.optimizer.ASGD.minimize

clear_grad(set_to_zero=True)
''''''''''''''''''''''''''''''''''''''''

.. note::

  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**参数**

    - **set_to_zero** (bool，可选) - 是否将梯度置零。若为 False，则删除梯度。默认值为 True。

**代码示例**

COPY-FROM: paddle.optimizer.ASGD.clear_grad

get_lr()
'''''''''

.. note::

  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

获取当前步骤的学习率。当不使用_LRScheduler 时，每次调用的返回值都相同，否则返回当前步骤的学习率。

**返回**

float，当前步骤的学习率。


**代码示例**

COPY-FROM: paddle.optimizer.ASGD.get_lr
