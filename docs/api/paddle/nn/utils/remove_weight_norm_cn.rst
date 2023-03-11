.. _cn_api_nn_cn_remove_weight_norm:

remove_weight_norm
-------------------------------

.. py:function:: paddle.nn.utils.remove_weight_norm(layer, name='weight')

移除传入 ``layer`` 中的权重归一化。

参数
::::::::::::

   - **layer** (Layer) - 要移除权重归一化的层。
   - **name** (str，可选) - 权重参数的名字。默认值为 ``weight``。

返回
::::::::::::

   ``Layer``，移除权重归一化 hook 之后的层。

代码示例
::::::::::::

COPY-FROM: paddle.nn.utils.remove_weight_norm
