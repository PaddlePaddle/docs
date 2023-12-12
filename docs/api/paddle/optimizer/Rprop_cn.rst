.. _cn_api_paddle_optimizer_Rprop:

Rprop
-------------------------------

.. py:class:: paddle.optimizer.Rprop(learning_rate=0.001, learning_rate_range=(1e-5, 50), parameters=None, etas=(0.5, 1.2), grad_clip=None, name=None)

    
.. note::
此优化器仅适用于 full-batch 训练。

Rprop算法的优化器。有关详细信息，请参阅：

`A direct adaptive method for faster backpropagation learning : The RPROP algorithm <https://ieeexplore.ieee.org/document/298623>`_ 。


.. math::

    \begin{aligned}
        &\hspace{0mm} For\ all\ weights\ and\ biases\{                                                                                                  \\
        &\hspace{5mm} \textbf{if} \: (\frac{\partial E}{\partial w_{ij}}(t-1)*\frac{\partial E}{\partial w_{ij}}(t)> 0)\ \textbf{then} \: \{            \\
        &\hspace{10mm} learning\_rate_{ij}(t)=\mathrm{minimum}(learning\_rate_{ij}(t-1)*\eta^{+},learning\_rate_{max})                                  \\
        &\hspace{10mm} \Delta w_{ij}(t)=-sign(\frac{\partial E}{\partial w_{ij}}(t))*learning\_rate_{ij}(t)                                             \\
        &\hspace{10mm} w_{ij}(t+1)=w_{ij}(t)+\Delta w_{ij}(t)                                                                                           \\
        &\hspace{5mm} \}                                                                                                                                \\
        &\hspace{5mm} \textbf{else if} \: (\frac{\partial E}{\partial w_{ij}}(t-1)*\frac{\partial E}{\partial w_{ij}}(t)< 0)\ \textbf{then} \: \{       \\
        &\hspace{10mm} learning\_rate_{ij}(t)=\mathrm{maximum}(learning\_rate_{ij}(t-1)*\eta^{-},learning\_rate_{min})                                  \\
        &\hspace{10mm} w_{ij}(t+1)=w_{ij}(t)                                                                                                            \\
        &\hspace{10mm} \frac{\partial E}{\partial w_{ij}}(t)=0                                                                                          \\
        &\hspace{5mm} \}                                                                                                                                \\
        &\hspace{5mm} \textbf{else if} \: (\frac{\partial E}{\partial w_{ij}}(t-1)*\frac{\partial E}{\partial w_{ij}}(t)= 0)\ \textbf{then} \: \{       \\
        &\hspace{10mm} \Delta w_{ij}(t)=-sign(\frac{\partial E}{\partial w_{ij}}(t))*learning\_rate_{ij}(t)                                             \\
        &\hspace{10mm} w_{ij}(t+1)=w_{ij}(t)+\Delta w_{ij}(t)                                                                                           \\
        &\hspace{5mm} \}                                                                                                                                \\
        &\hspace{0mm} \}                                                                                                                                \\
    \end{aligned}


参数
::::::::::::

    - **learning_rate** (float|_LRScheduleri，可选) - 初始学习率，用于参数更新的计算。可以是一个浮点型值或者一个_LRScheduler 类，默认值为 0.001。
    - **learning_rate_range** (tuple，可选) - 学习率的范围。学习率不能小于元组的第一个元素；学习率不能大于元组的第二个元素。
    - **parameters** (list，可选) - 指定优化器需要优化的参数。在动态图模式下必须提供该参数；在静态图模式下默认值为 None，这时所有的参数都将被优化。
    - **etas** (tuple，可选) - 用于更新学习率的元组。元组的第一个元素是乘法递减因子；元组的第二个元素是乘法增加因子。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值为 None，此时将不进行梯度裁剪。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


代码示例
::::::::::::

COPY-FROM: paddle.optimizer.Rprop


方法
::::::::::::
step()
'''''''''

.. note::

  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

执行一次优化器并进行参数更新。

**返回**

无。



COPY-FROM: paddle.optimizer.Rprop.step

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

COPY-FROM: paddle.optimizer.Rprop.minimize

clear_grad()
'''''''''

.. note::

  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。


清除需要优化的参数的梯度。

**代码示例**

COPY-FROM: paddle.optimizer.Rprop.clear_grad

get_lr()
'''''''''

.. note::

  该 API 只在 `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ 模式下生效。

获取当前步骤的学习率。当不使用_LRScheduler 时，每次调用的返回值都相同，否则返回当前步骤的学习率。

**返回**

float，当前步骤的学习率。


**代码示例**

COPY-FROM: paddle.optimizer.Rprop.get_lr
