.. _cn_api_tensor_ones_like:

ones_like
-------------------------------

.. py:function:: paddle.ones_like(x, dtype=None, name=None)


返回一个和输入参数 ``x`` 具有相同形状的数值都为1的 Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同，如果 ``dtype`` 为None，则输出Tensor的数据类型与 ``x`` 相同。

参数
::::::::::
    - **x** (Tensor) – 输入的Tensor，数据类型可以是 bool，float16，float32，float64，int32，int64。
    - **dtype** (str|np.dtype， 可选) - 输出 Tensor 的数据类型，支持 bool，float16, float32，float64，int32，int64。当该参数值为 None 时， 输出 Tensor 的数据类型与 ``x`` 相同。默认值为 None。
    - **name** (str， 可选) - 输出的名字。一般无需设置，默认值为 None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。
    
返回
::::::::::

Tensor：和 ``x`` 具有相同形状的数值都为1的 Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同。


代码示例
::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([1,2,3])
    out1 = paddle.ones_like(x) # [1., 1., 1.]
    out2 = paddle.ones_like(x, dtype='int32') # [1, 1, 1]
