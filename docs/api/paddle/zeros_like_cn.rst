.. _cn_api_tensor_zeros_like:

zeros_like
-------------------------------

.. py:function:: paddle.zeros_like(x, dtype=None, name=None)


返回一个和 ``x`` 具有相同的形状的全零Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同。

参数
::::::::::
    - **x** (Tensor) – 输入的多维Tensor，数据类型可以是bool，float16，float32，float64，int32，int64。输出Tensor的形状和 ``x`` 相同。如果 ``dtype`` 为None，则输出Tensor的数据类型与 ``x`` 相同。
    - **dtype** (str|np.dtype，可选) - 输出Tensor的数据类型，支持bool，float16，float32，float64，int32，int64。当该参数值为None时，输出Tensor的数据类型与 ``x`` 相同。默认值为None.
    - **name** (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。
    
返回
::::::::::
    Tensor：和 ``x`` 具有相同的形状全零Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同。


代码示例
::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([1, 2, 3])
    out1 = paddle.zeros_like(x) # [0., 0., 0.]
    out2 = paddle.zeros_like(x, dtype='int32') # [0, 0, 0]
