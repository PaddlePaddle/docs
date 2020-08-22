.. _cn_api_paddle_tensor_linalg_dot:

dot
-------------------------------

.. py:function:: paddle.tensor.linalg.dot(x, y, name=None)

:alias_main: paddle.dot
:alias: paddle.dot, paddle.tensor.dot, paddle.tensor.linalg.dot



该OP计算向量的内积

.. note::

   仅支持1维Tensor(向量).

参数：
        - **x** （Variable）- 1维 ``Tensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
        - **y** （Variable）- 1维 ``Tensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
        - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。


返回：  ``Tensor`` ，数据类型与 ``x`` 相同。

返回类型：        Variable。

**代码示例**

..  code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()
    x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
    y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
    x = paddle.imperative.to_variable(x_data)
    y = paddle.imperative.to_variable(y_data)
    z = paddle.dot(x, y)
    print(z.numpy())
