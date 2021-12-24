.. _cn_api_nn_utils_parameters_to_vector:

parameters_to_vector
-------------------------------

.. py:function:: paddle.nn.utils.parameters_to_vector(parameters, name=None)

将输入的多个parameter展平并连接为1个1-D Tensor。

参数
:::::::::
    - parameters (Iterable[Tensor]) - 可迭代的多个parameter。parameter为Layer中可训练的Tensor。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ，多个parameter展平并连接的1-D Tensor

代码示例
:::::::::

.. code-block:: python

    import paddle
    linear = paddle.nn.Linear(10, 15)

    paddle.nn.utils.parameters_to_vector(linear.parameters())
    # 1-D Tensor: [165]
