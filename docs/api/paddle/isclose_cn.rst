.. _cn_api_tensor_isclose:

isclose
-------------------------------

.. py:function:: paddle.isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)

逐个检查x和y的所有元素是否均满足如下条件：

..  math::
    \left| x - y \right| \leq atol + rtol \times \left| y \right|

该API的行为类似于 :math:`numpy.isclose` ，即逐个比较两个Tensor的所有元素是否在一定容忍误差范围内视为相等。

参数:
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64。
    - **y** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64。
    - **rtol** (float，可选) - 相对容忍误差，默认值为1e-5。
    - **atol** (float，可选) - 绝对容忍误差，默认值为1e-8。
    - **equal_nan** (bool，可选) - 如果设置为True，则两个NaN数值将被视为相等，默认值为False。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：计算得到的布尔类型Tensor。

**代码示例**:

.. code-block:: python

    import paddle
    x = paddle.to_tensor([10000., 1e-07])
    y = paddle.to_tensor([10000.1, 1e-08])
    result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                            equal_nan=False, name="ignore_nan")
    np_result1 = result1.numpy()
    # [True, False]
    result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                                equal_nan=True, name="equal_nan")
    np_result2 = result2.numpy()
    # [True, False]
    x = paddle.to_tensor([1.0, float('nan')])
    y = paddle.to_tensor([1.0, float('nan')])
    result1 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                            equal_nan=False, name="ignore_nan")
    np_result1 = result1.numpy()
    # [True, False]
    result2 = paddle.isclose(x, y, rtol=1e-05, atol=1e-08,
                                equal_nan=True, name="equal_nan")
    np_result2 = result2.numpy()
    # [True, True]