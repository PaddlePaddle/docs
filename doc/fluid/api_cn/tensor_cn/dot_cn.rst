.. _cn_api_paddle_tensor_linalg_dot:

dot
-------------------------------

.. py:function:: paddle.tensor.linalg.dot(x, y, name=None)

:alias_main: paddle.dot
:alias: paddle.dot,paddle.tensor.dot,paddle.tensor.linalg.dot



该OP计算向量的内积

.. note::
   仅支持1维Tensor(向量).

参数：
        - **x** （Variable）- 1维 ``Tensor`` 或 ``LoDTensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **y** （Variable）- 1维 ``Tensor`` 或 ``LoDTensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。


返回：  ``Tensor`` 或 ``LoDTensor`` ，数据类型与 ``x`` 相同。

返回类型：        Variable。

**代码示例**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
      x = fluid.dygraph.to_variable(np.random.uniform(0.1, 1, [10]).astype(np.float32))
      # [0.94936 0.119406 0.696538 0.611788 0.864026 0.858593 0.198292 0.782601 0.370953 0.97247]
      y = fluid.dygraph.to_variable(np.random.uniform(1, 3, [10]).astype(np.float32))
      # [2.94794 1.24812 1.91361 2.56719 2.92583 1.83007 1.23439 2.70771 2.04935 2.66508]
      z = paddle.dot(x, y)
      # [15.666201]


