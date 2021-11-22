.. _cn_api_tensor_zeropad2d:

zeropad2d
-------------------------------
.. py:function:: paddle.zeropad2d(x, padding, data_format="NCHW", name=None)

该OP返回一个按照 ``padding`` 属性对 ``x`` 进行零填充的Tensor，数据类型与 ``x`` 相同。

参数
::::::::::
    - **x** (Tensor) - Tensor，format可以为 ``'NCHW'``, ``'NHWC'`` ，默认值为 ``'NCHW'``，数据类型支持float16, float32, float64, int32, int64。
    - **padding** (Tensor | List[int] | Tuple[int]) - 填充大小。pad的格式为[pad_left, pad_right, pad_top, pad_bottom]；
    - **data_format** (str)  - 指定 ``x`` 的format，可为 ``'NCHW'``, ``'NHWC'``, 默认值为 ``'NCHW'``。
    - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。

返回
::::::::::
    Tensor：对 ``x`` 进行 ``'pad'`` 的结果，数据类型和 ``x`` 相同。

代码示例
::::::::::

.. code-block:: python

    import paddle
    import numpy as np

    x_shape = (1, 1, 2, 3)
    x = paddle.arange(np.prod(x_shape), dtype="float32").reshape(x_shape) + 1
    y = paddle.zeropad2d(x, [1, 2, 1, 1])

    # [[[[0. 0. 0. 0. 0. 0.]
    #    [0. 1. 2. 3. 0. 0.]
    #    [0. 4. 5. 6. 0. 0.]
    #    [0. 0. 0. 0. 0. 0.]]]]


