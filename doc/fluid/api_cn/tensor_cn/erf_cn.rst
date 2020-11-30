.. _cn_api_tensor_erf:

erf
-------------------------------

.. py:function:: paddle.erf(x, name=None)



逐元素计算 Erf 激活函数。更多细节请参考 `Error function <https://en.wikipedia.org/wiki/Error_function>`_ 。


.. math::
    out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

参数：
    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为： float16, float32, float64。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回：
    - Tensor，对输入x进行erf激活后的Tensor，形状、数据类型与输入 x 一致。


**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
    out = paddle.erf(x)
    print(out)
    # Tensor(shape=[4], dtype=float32, place=CPUPlace, stop_gradient=True,
    #        [-0.42839241, -0.22270259,  0.11246292,  0.32862678])
