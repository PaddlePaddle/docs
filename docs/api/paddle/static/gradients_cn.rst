.. _cn_api_paddle_static_gradients:

gradients
-------------------------------


.. py:function:: paddle.static.gradients(targets, inputs, target_gradients=None, no_grad_set=None)




将目标 Tensor 的梯度反向传播到输入 Tensor。

参数
::::::::::::

  - **targets** (Tensor|list[Tensor]) – 目标 Tensor 或包含 Tensor 的列表。
  - **inputs** (Tensor|list[Tensor]) – 输入 Tensor 或包含 Tensor 的列表。
  - **target_gradients** (Tensor|list[Tensor]，可选) – 目标的梯度 Tensor，应与目标 Tensor 的形状相同；如果设置为 None，则以 1 初始化所有梯度 Tensor。
  - **no_grad_set** (set[Tensor|str]，可选) – 在 `block0` ( :ref:`api_guide_Block` ) 中要忽略梯度的 Tensor 的名字的集合。所有的 :ref:`api_guide_Block` 中带有 ``stop_gradient = True`` 的所有 Tensor 的名字都会被自动添加到此集合中。如果该参数不为 ``None``，则会将该参数集合的内容添加到默认的集合中。默认值为 ``None`` 。


返回
::::::::::::
list[Tensor]，包含与输入对应的梯度。如果一个输入不影响目标函数，则对应的梯度 Tensor 为 None 。


代码示例
::::::::::::

COPY-FROM: paddle.static.gradients
