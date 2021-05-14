.. _cn_api_tensor_sign:

sign
-------------------------------

.. py:function:: paddle.sign(x, name=None)

此OP对输入x中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零。

参数：
    - **x** (Tensor) – 进行正负值判断的多维Tensor，数据类型为 float16， float32或float64。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出正负号Tensor，数据的shape大小及数据类型和输入 ``x`` 一致。

返回类型：Tensor

**代码示例**

..  code-block:: python

    import numpy as np
    import paddle

    data = np.array([3.0, 0.0, -2.0, 1.7], dtype='float32')
    x = paddle.to_tensor(data)
    out = paddle.sign(x=x)
    print(out)  # [1.0, 0.0, -1.0, 1.0]

