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

.. note:: 
    ``numpy.where(condition)`` 功能与 ``paddle.nonzero(condition, as_tuple=True)`` 相同。

参数：
    - **condition** （Tensor）- 选择 ``x`` 或 ``y`` 元素的条件 。
    - **x** （Tensor，Scalar，可选）- 多维 ``Tensor`` 或 ``Scalar``，数据类型为 ``float32`` 或 ``float64`` 或 ``int32`` 或 ``int64`` 。``x`` 和 ``y`` 必须都给出或者都不给出。
    - **y** （Tensor，Scalar，可选）- 多维 ``Tensor`` 或 ``Scalar``，数据类型为 ``float32`` 或 ``float64`` 或 ``int32`` 或 ``int64`` 。``x`` 和 ``y`` 必须都给出或者都不给出。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：数据类型与 ``x`` 相同的 ``Tensor`` 。

返回类型：Tensor。


**代码示例：**

.. code-block:: python

          import paddle

          x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
          y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
          out = paddle.where(x>1, x, y)

          print(out)
          #out: [1.0, 1.0, 3.2, 1.2]

          out = paddle.where(x>1)
          print(out)
          #out: (Tensor(shape=[2, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
          #            [[2],
          #             [3]]),)