.. _cn_api_paddle_neg:

neg
-------------------------------

.. py:function:: paddle.neg(x, name=None)




计算输入 x 的相反数并返回。

.. math::
    out = -x

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：int8、int16、int32、int64、float32、float64。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
输出Tensor，与 ``x`` 维度相同、数据类型相同。

代码示例
:::::::::

.. code-block:: python

        import paddle
        
        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3], dtype='float32')
        res = paddle.neg(x)
        print(res)
        # [0.4, 0.2, -0.1, -0.3]
