.. _cn_api_paddle_optimizer_AdamW:

AdamW
-------------------------------

.. py:class:: paddle.optimizer.AdamW(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=None, weight_decay=0.01, lr_ratio=None, apply_decay_param_fun=None, grad_clip=None, lazy_mode=False, multi_precision=False, name=None)




AdamW 优化器出自 `DECOUPLED WEIGHT DECAY REGULARIZATION <https://arxiv.org/pdf/1711.05101.pdf>`_ ，用来解决 :ref:`Adam <cn_api_paddle_optimizer_Adam>` 优化器中 L2 正则化失效的问题。

其参数更新的计算公式如下：

.. math::
    \\t = t + 1
.. math::
    moment\_1\_out=\beta_1∗moment\_1+(1−\beta_1)∗grad
.. math::
    moment\_2\_out=\beta_2∗moment\_2+(1−\beta_2)∗grad*grad
.. math::
    learning\_rate=learning\_rate*\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
.. math::
    param\_out=param-learning\_rate*(\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} + \lambda * param)

相关论文：`Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

.. note::
  当前，AdamW 不支持稀疏参数优化。

参数
::::::::::::

    - **learning_rate** (float|_LRScheduler) - 学习率，用于参数更新的计算。可以是一个浮点型值或者一个_LRScheduler 类，默认值为 0.001。
    - **beta1** (float|Tensor，可选) - 一阶矩估计的指数衰减率，是一个 float 类型或者一个 shape 为[1]，数据类型为 float32 的 Tensor 类型。默认值为 0.9。
    - **beta2** (float|Tensor，可选) - 二阶矩估计的指数衰减率，是一个 float 类型或者一个 shape 为[1]，数据类型为 float32 的 Tensor 类型。默认值为 0.999。
    - **epsilon** (float，可选) - 保持数值稳定性的短浮点类型值，默认值为 1e-08。
    - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **weight_decay** (float|Tensor，可选) - 权重衰减系数，是一个 float 类型或者 shape 为[1]，数据类型为 float32 的 Tensor 类型。默认值为 0.01。
    - **lr_ratio** (function|None，可选) – 传入函数时，会为每个参数计算一个权重衰减系数，并使用该系数与学习率的乘积作为新的学习率。否则，使用原学习率。仅支持 GPU 设备，默认值为 None。
    - **apply_decay_param_fun** (function|None，可选)：传入函数时，只有可以使 apply_decay_param_fun(Tensor.name)==True 的 Tensor 会进行 weight decay 更新。只有在想要指定特定需要进行 weight decay 更新的参数时使用。默认值为 None。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **lazy_mode** （bool，可选） - 设为 True 时，仅更新当前具有梯度的元素。官方 Adam 算法有两个移动平均累加器（moving-average accumulators）。累加器在每一步都会更新。在密集模式和稀疏模式下，两条移动平均线的每个元素都会更新。如果参数非常大，那么更新可能很慢。lazy mode 仅更新当前具有梯度的元素，所以它会更快。但是这种模式与原始的算法有不同的描述，可能会导致不同的结果，默认为 False。
    - **multi_precision** (bool，可选) – 在基于 GPU 设备的混合精度训练场景中，该参数主要用于保证梯度更新的数值稳定性。设置为 True 时，优化器会针对 FP16 类型参数保存一份与其值相等的 FP32 类型参数备份。梯度更新时，首先将梯度类型提升到 FP32，然后将其更新到 FP32 类型参数备份中。最后，更新后的 FP32 类型值会先转换为 FP16 类型，再赋值给实际参与计算的 FP16 类型参数。默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


代码示例
::::::::::::

COPY-FROM: paddle.optimizer.AdamW

方法
::::::::::::
step()
'''''''''

.. note::
  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

执行一次优化器并进行参数更新。

**返回**

无。


**代码示例**

COPY-FROM: paddle.optimizer.AdamW.step

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

为网络添加反向计算过程，并根据反向计算所得的梯度，更新 parameters 中的 Parameters，最小化网络损失值 loss。

**参数**

    - **loss** (Tensor) - 需要最小化的损失值变量。
    - **startup_program** (Program，可选) - 用于初始化 parameters 中参数的 :ref:`cn_api_paddle_static_Program`，默认值为 None，此时将使用 :ref:`cn_api_paddle_static_default_startup_program` 。
    - **parameters** (list，可选) - 待更新的 Parameter 或者 Parameter.name 组成的列表，默认值为 None，此时将更新所有的 Parameter。
    - **no_grad_set** (set，可选) - 不需要更新的 Parameter 或者 Parameter.name 组成的集合，默认值为 None。

**返回**

tuple(optimize_ops, params_grads)，其中 optimize_ops 为参数优化 OP 列表；param_grads 为由(param, param_grad)组成的列表，其中 param 和 param_grad 分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为 True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

COPY-FROM: paddle.optimizer.AdamW.minimize

clear_grad()
'''''''''

.. note::
  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

COPY-FROM: paddle.optimizer.AdamW.clear_grad

set_lr(value)
'''''''''

.. note::
  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

手动设置当前 ``optimizer`` 的学习率。当使用_LRScheduler 时，无法使用该 API 手动设置学习率，因为这将导致冲突。

**参数**

    value (float) - 需要设置的学习率的值。

**返回**

无。

**代码示例**

COPY-FROM: paddle.optimizer.AdamW.set_lr

set_lr_scheduler(scheduler)
'''''''''

.. note::
  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

手动设置当前 ``optimizer`` 的学习率为 LRScheduler 类。

**参数**

    scheduler (LRScheduler) - 需要设置的学习率的 LRScheduler 类。

**返回**

无。

**代码示例**

.. code-block:: python

    import paddle
    linear = paddle.nn.Linear(10, 10)
    adam = paddle.optimizer.AdamW(weight_decay=0.01,
                                 learning_rate=0.1, parameters=linear.parameters())
    # set learning rate manually by class LRScheduler
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2,4,6], gamma=0.8)
    adam.set_lr_scheduler(scheduler)
    lr = adam.get_lr()
    print("current lr is {}".format(lr))
    #    current lr is 0.5
    # set learning rate manually by another LRScheduler
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.1, step_size=5, gamma=0.6)
    adam.set_lr_scheduler(scheduler)
    lr = adam.get_lr()
    print("current lr is {}".format(lr))
    #    current lr is 0.1

get_lr()
'''''''''

.. note::
  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

获取当前步骤的学习率。当不使用_LRScheduler 时，每次调用的返回值都相同，否则返回当前步骤的学习率。

**返回**

float，当前步骤的学习率。


**代码示例**

COPY-FROM: paddle.optimizer.AdamW.get_lr
