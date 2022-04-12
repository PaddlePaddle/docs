.. _cn_api_tensor_frac:

frac
-------------------------------

.. py:function:: paddle.frac(x, name=None)


得到输入 `Tensor` 的小数部分。


参数
:::::::::
    - **x** (Tensor) : 输入变量，类型为 Tensor, 支持int32、int64、float32、float64数据类型。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
    - Tensor (Tensor)，输入矩阵只保留小数部分的结果。


代码示例
:::::::::

.. code-block:: python

    import paddle

     = paddle.rand([3,3],'float32')
    print(x)
    # Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [[5.29032421, 7.70876980, 5.14640331],
    #         [2.30558801, 7.60625172, 2.57993436],
    #         [1.53053904, 1.51977015, 2.96169519]])

    output = paddle.frac(x)
    print(output)
    # Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [[0.29032421, 0.70876980, 0.14640331],
    #         [0.30558801, 0.60625172, 0.57993436],
    #         [0.53053904, 0.51977015, 0.96169519]])