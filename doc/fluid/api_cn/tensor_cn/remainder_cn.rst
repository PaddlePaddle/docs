.. _cn_api_tensor_remainder:

remainder
-------------------------------

.. py:function:: paddle.remainder(x, y, name=None)

逐元素取模算子。公式为：

.. math::
        out = x \% y

**注意**:
        ``paddle.remainder`` 支持广播。关于广播规则，请参考 :ref:`use_guide_broadcasting`

参数：
        - x（Tensor）- 多维Tensor。数据类型为float32 、float64、int32或int64。
        - y（Tensor）- 多维Tensor。数据类型为float32 、float64、int32或int64。
        - name（str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回：
        多维Tensor。存储计算的结果，数据类型与 ``x`` 相同，维度为广播后的形状。

**代码示例**

..  code-block:: python

        import paddle

        x = paddle.to_tensor([2, 3, 8, 7])
        y = paddle.to_tensor([1, 5, 3, 3])
        z = paddle.remainder(x, y)
        print(z)  # [0, 3, 2, 1]