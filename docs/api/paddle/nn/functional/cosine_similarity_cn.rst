.. _cn_api_paddle_nn_cosine_similarity:

cosine_similarity
-------------------------------

.. py:function:: paddle.nn.functional.cosine_similarity(x1, x2, axis=1, eps=1e-8)

该OP用于计算x1与x2沿axis维度的余弦相似度。

参数：
  - **x1** (Tensor) - Tensor，数据类型支持float32, float64。
  - **x2** (Tensor) - Tensor，数据类型支持float32, float64。
  - **axis** (int) - 指定计算的维度，会在该维度上计算余弦相似度，默认值为1。
  - **eps** (float) - 很小的值，防止计算时分母为0，默认值为1e-8。
  
  
返回： 余弦相似度的计算结果，数据类型与x1, x2相同。

返回类型：Tensor



**代码示例：**

.. code-block:: python

    import paddle
    import paddle.nn as nn
    import numpy as np

    np.random.seed(0)
    x1 = np.random.rand(2,3)
    x2 = np.random.rand(2,3)
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    result = paddle.nn.functional.cosine_similarity(x1, x2, axis=0)
    print(result)
    # [0.99806249 0.9817672  0.94987036]



