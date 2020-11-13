.. _cn_api_tensor_where:

where
-------------------------------

.. py:function:: paddle.where(condition, x, y, name=None)




该OP返回一个根据输入 ``condition``, 选择 ``x`` 或 ``y`` 的元素组成的多维 ``Tensor``  ：

.. math::
      Out_i =
      \left\{
      \begin{aligned}
      &X_i, & & if \ cond_i \ is \ True \\
      &Y_i, & & if \ cond_i \ is \ False \\
      \end{aligned}
      \right.

参数：
    - **condition** （Tensor）- 选择 ``x`` 或 ``y`` 元素的条件 。
    - **x** （Tensor）- 多维 ``Tensor`` ，数据类型为 ``float32`` 或 ``float64`` 或 ``int32`` 或 ``int64`` 。
    - **y** （Tensor）- 多维 ``Tensor`` ，数据类型为 ``float32`` 或 ``float64`` 或 ``int32`` 或 ``int64`` 。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：数据类型与 ``x`` 相同的 ``Tensor`` 。

返回类型：Tensor。


**代码示例：**

.. code-block:: python

          import paddle
          import numpy as np
          import paddle.fluid as fluid

          x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
          y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
          out = paddle.where(x>1, x, y)

          print(out)
          #out: [1.0, 1.0, 3.2, 1.2]
