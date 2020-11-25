.. _cn_api_tensor_flip:

flip
-------------------------------

.. py:function:: paddle.flip(x, axis, name=None):




该OP沿指定轴反转n维tensor.

参数：
    - **x** (Tensor) - 输入张量。维度为多维，数据类型为bool, int32, int64, float32或float64。
    - **axis** (list) - 需要翻转的轴。当 ``axis[i] < 0`` 时，实际的计算维度为 ndim(x) + axis[i]，其中i为axis的索引。
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。

返回：在指定axis上翻转后的张量，与输入x数据类型相同。

返回类型：Tensor，与输入x数据类型相同。

抛出异常：
    - ``TypeError`` - 当输出 ``out`` 和输入 ``x`` 数据类型不一致时候。
    - ``ValueError`` - 当参数  ``axis`` 不合法时。

**代码示例1**：

.. code-block:: python

    import paddle
    import numpy as np
    
    image_shape=(3, 2, 2)
    x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
    x = x.astype('float32')
    img = paddle.imperative.to_variable(x)
    out = paddle.flip(img, [0,1])
    print(out) # [[[10,11][8, 9]],[[6, 7],[4, 5]] [[2, 3],[0, 1]]]

