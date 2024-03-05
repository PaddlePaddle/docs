.. _cn_api_paddle_incubate_optimizer_LBFGS:

LBFGS
-------------------------------

.. py:class:: paddle.incubate.optimizer.LBFGS(lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None, parameters=None, weight_decay=None, grad_clip=None, name=None)

``LBFGS`` 使用 L-BFGS 方法对参数进行优化更新，使得 loss 值最小。

L-BFGS 是限制内存的 BFGS 方法，适用于海森矩阵为稠密矩阵、内存开销较大场景。BFGS 参考 :ref:`cn_api_paddle_incubate_optimizer_functional_minimize_bfgs` .

LBFGS 具体原理参考书籍 Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp179: Algorithm 7.5 (L-BFGS).


使用方法
:::::::::
- LBFGS 优化器此实现为类形式，与 Paddle 现有 SGD、Adam 优化器相似。
  通过调用 backward()计算梯度，并使用 step(closure)更新网络参数，其中 closure 为需要优化的闭包函数。


.. warning::
  该 API 目前为 Beta 版本，函数签名在未来版本可能发生变化。

.. note::
  当前仅支持动态图模式下使用。


参数
:::::::::
    - **lr** (float，可选) - 学习率，用于参数更新的计算，默认值：1.0。
    - **max_iter** (Scalar，可选) - 每个优化单步的最大迭代次数，默认值：20。
    - **max_eval** (Scalar，可选) - 每次优化单步中函数计算的最大数量，默认值：max_iter * 1.25。
    - **tolerance_grad** (float，可选) - 当梯度的范数小于该值时，终止迭代。当前使用正无穷范数。默认值：1e-5。
    - **tolerance_change** (float，可选) - 当函数值/x 值/其他参数 两次迭代的改变量小于该值时，终止迭代。默认值：1e-9。
    - **history_size** (Scalar，可选) - 指定储存的向量对{si,yi}数量。默认值：100。
    - **line_search_fn** (str，可选) - 指定要使用的线搜索方法，目前支持值为'strong wolfe'方法。默认值：'None'。
    - **parameters** (list，可选) - 指定优化器需要优化的参数，在动态图模式下必须提供该参数。默认值：None。
    - **weight_decay** (float|WeightDecayRegularizer，可选) - 正则化方法。可以是 float 类型的 L2 正则化系数或者正则化策略：:ref:`cn_api_paddle_regularizer_L1Decay` 、
      :ref:`cn_api_paddle_regularizer_L2Decay` 。如果一个参数已经在 :ref:`cn_api_paddle_ParamAttr` 中设置了正则化，这里的正则化设置将被忽略；
      如果没有在 :ref:`cn_api_paddle_ParamAttr` 中设置正则化，这里的设置才会生效。默认值为 None，表示没有正则化。
    - **grad_clip** (GradientClipBase，可选) – 梯度裁剪的策略，支持三种裁剪策略：:ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_paddle_nn_ClipGradByGlobalNorm>` 、 :ref:`paddle.nn.ClipGradByNorm <cn_api_paddle_nn_ClipGradByNorm>` 、 :ref:`paddle.nn.ClipGradByValue <cn_api_paddle_nn_ClipGradByValue>` 。
      默认值：None，此时将不进行梯度裁剪。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值：None。

返回
:::::::::
    - loss (Tensor)：迭代终止位置的损失函数值。

代码示例
::::::::::

COPY-FROM: paddle.incubate.optimizer.LBFGS
