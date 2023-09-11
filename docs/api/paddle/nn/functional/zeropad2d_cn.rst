.. _cn_api_paddle_nn_functional_zeropad2d:

zeropad2d
-------------------------------
.. py:function:: paddle.nn.functional.zeropad2d(x, padding, data_format="NCHW", name=None)

返回一个按照 ``padding`` 属性对 ``x`` 进行零填充的 Tensor，数据类型与 ``x`` 相同。

参数
::::::::::
    - **x** (Tensor) - Tensor，format 可以为 ``'NCHW'``, ``'NHWC'``，默认值为 ``'NCHW'``，数据类型支持 float16, float32, float64, int32, int64。
    - **padding** (Tensor | List[int] | Tuple[int]) - 填充大小。pad 的格式为[pad_left, pad_right, pad_top, pad_bottom]；
    - **data_format** (str，可选)  - 指定 ``x`` 的 format，可为 ``'NCHW'``, ``'NHWC'``，默认值为 ``'NCHW'``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    Tensor：对 ``x`` 进行 ``'pad'`` 的结果，数据类型和 ``x`` 相同。

代码示例
::::::::::

COPY-FROM: paddle.nn.functional.zeropad2d
