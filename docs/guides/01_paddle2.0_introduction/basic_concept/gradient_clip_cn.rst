梯度裁剪方式介绍
====================

神经网络是通过梯度下降来进行网络学习，随着网络层数的增加，"梯度爆炸"的问题可能会越来越明显。例如：在梯度反向传播中，如果每一层的输出相对输入的偏导 > 1，随着网络层数的增加，梯度会越来越大，则有可能发生 "梯度爆炸"。

如果发生了 "梯度爆炸"，在网络学习过程中会直接跳过最优解，所以有必要进行梯度裁剪，防止网络在学习过程中越过最优解。

Paddle提供了三种梯度裁剪方式：

一、设定范围值裁剪
--------------------

设定范围值裁剪：将参数的梯度限定在一个范围内，如果超出这个范围，则进行裁剪。

使用方式：需要创建一个 :ref:`paddle.nn.ClipGradByValue <cn_api_fluid_clip_ClipGradByValue>` 类的实例，然后传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

**1. 全部参数裁剪（默认）**

默认情况下，会裁剪优化器中全部参数的梯度：

.. code:: ipython3

    import paddle

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByValue(min=-1, max=1)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

如果仅需裁剪部分参数，用法如下：

**2. 部分参数裁剪**

部分参数裁剪需要设置参数的 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` ，其中的 ``need_clip`` 默认为True，表示需要裁剪，如果设置为False，则不会裁剪。

例如：仅裁剪 `linear` 中 `weight` 的梯度，则需要在创建 `linear` 层时设置 `bias_attr` 如下：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10，bias_attr=paddle.ParamAttr(need_clip=False))

二、通过L2范数裁剪
--------------------

通过L2范数裁剪：梯度作为一个多维Tensor，计算其L2范数，如果超过最大值则按比例进行裁剪，否则不裁剪。

使用方式：需要创建一个 :ref:`paddle.nn.ClipGradByNorm <cn_api_fluid_clip_ClipGradByNorm>` 类的实例，然后传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

裁剪公式如下：

.. math::

  Out=
  \left\{
  \begin{aligned}
  &  X & & if (norm(X) \leq clip\_norm)\\
  &  \frac{clip\_norm∗X}{norm(X)} & & if (norm(X) > clip\_norm) \\
  \end{aligned}
  \right.


其中 :math:`norm（X）` 代表 :math:`X` 的L2范数

.. math::
  \\norm(X) = (\sum_{i=1}^{n}|x_i|^2)^{\frac{1}{2}}\\

**1. 全部参数裁剪（默认）**

默认情况下，会裁剪优化器中全部参数的梯度：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

如果仅需裁剪部分参数，用法如下：

**2. 部分参数裁剪**

部分参数裁剪的设置方式与上面一致，也是通过设置参数的 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` ，其中的 ``need_clip`` 默认为True，表示需要裁剪，如果设置为False，则不会裁剪。

例如：仅裁剪 `linear` 中 `bias` 的梯度，则需要在创建 `linear` 层时设置 `weight_attr` 如下：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10, weight_attr=paddle.ParamAttr(need_clip=False))


三、通过全局L2范数裁剪
--------------------

将优化器中全部参数的梯度组成向量，对该向量求解L2范数，如果超过最大值则按比例进行裁剪，否则不裁剪。

使用方式：需要创建一个 :ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_fluid_clip_ClipGradByGlobalNorm>` 类的实例，然后传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

裁剪公式如下：

.. math::

  Out[i]=
  \left\{
  \begin{aligned}
  &  X[i] & & if (global\_norm \leq clip\_norm)\\
  &  \frac{clip\_norm∗X[i]}{global\_norm} & & if (global\_norm > clip\_norm) \\
  \end{aligned}
  \right.


其中：

.. math::  
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(norm(X[i]))^2}\\


其中 :math:`norm（X）` 代表 :math:`X` 的L2范数

**1. 全部参数裁剪（默认）**

默认情况下，会裁剪优化器中全部参数的梯度：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByGloabalNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

如果仅需裁剪部分参数，用法如下：

**2. 部分参数裁剪**

部分参数裁剪的设置方式与上面一致，也是通过设置参数的 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` ，其中的 ``need_clip`` 默认为True，表示需要裁剪，如果设置为False，则不会裁剪。可参考上面的示例代码进行设置。
