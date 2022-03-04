.. _cn_api_nn_utils_vector_to_parameters:

vector_to_parameters
-------------------------------

.. py:function:: paddle.nn.utils.vector_to_parameters(vec, parameters, name=None)

将1个1-D Tensor按顺序切分给输入的多个parameter。

参数
:::::::::
    - vec (Tensor) - 一个1-D Tensor。
    - parameters (Iterable[Tensor]) - 可迭代的多个parameter。parameter为Layer中可训练的Tensor。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
无

代码示例
:::::::::

.. code-block:: python

    import paddle
    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(3.))
    linear1 = paddle.nn.Linear(10, 15, weight_attr)

    vec = paddle.nn.utils.parameters_to_vector(linear1.parameters())

    linear2 = paddle.nn.Linear(10, 15)
    # copy weight of linear1 to linear2
    paddle.nn.utils.vector_to_parameters(vec, linear2.parameters())
    # weight: Tensor(shape=[10, 15], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
    #                 [[3. , ..., 3. ],
    #                  [..., ..., ...],
    #                  [3. , ..., 3. ]])
    