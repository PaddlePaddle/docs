.. _cn_api_paddle_nn_functional_cosine_similarity:

cosine_similarity
-------------------------------

.. py:function:: paddle.nn.functional.cosine_similarity(x1, x2, axis=1, eps=1e-8)

用于计算 x1 与 x2 沿 axis 维度的余弦相似度。

参数
::::::::::::

  - **x1** (Tensor) - Tensor，数据类型支持 float32, float64。
  - **x2** (Tensor) - Tensor，数据类型支持 float32, float64。
  - **axis** (int，可选) - 指定计算的维度，会在该维度上计算余弦相似度，默认值为 1。
  - **eps** (float，可选) - 很小的值，防止计算时分母为 0，默认值为 1e-8。


返回
::::::::::::
Tensor，余弦相似度的计算结果，数据类型与 x1, x2 相同。



代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.cosine_similarity
