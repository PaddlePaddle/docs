.. _cn_api_tensor_isnan:

isnan
-----------------------------

.. py:function:: paddle.isnan(x, name=None)

返回输入tensor的每一个值是否为 `+/-NaN` 。

参数
:::::::::
    - **x** (Tensor): 输入的 `Tensor` ，数据类型为：float16、float32、float64、int32、int64。
    - **name** (str, 可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor``, 每个元素是一个bool值，表示输入 `x` 的每个元素是否为 `+/-NaN` 。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    x_np = np.array([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
    x = paddle.to_tensor(x_np)
    out = paddle.isnan(x)
    print(out)  # [False, False, False, False, False, True, True]
