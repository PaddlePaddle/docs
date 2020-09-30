.. _cn_api_tensor_linalg_cross:

cross
-------------------------------

.. py:function:: paddle.cross(x, y, axis=None, name=None)


计算张量 ``x`` 和 ``y`` 在 ``axis`` 维度上的向量积（叉积）。 

``x`` 和 ``y`` 必须有相同的形状，且指定的 ``axis`` 的长度必须为3. 如果未指定 ``axis`` ，默认选取第一个长度为3的 ``axis`` .
        
参数
:::::::::
    - x (Tensor) – 第一个输入张量。
    - y (Tensor) – 第二个输入张量。
    - axis (int, 可选) – 沿着此维进行向量积操作。默认选取第一个长度为3的 ``axis`` .
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``, 向量积的结果。

代码示例
::::::::::

.. code-block:: python

        import paddle

        x = paddle.to_tensor([[1.0, 1.0, 1.0],
                                [2.0, 2.0, 2.0],
                                [3.0, 3.0, 3.0]])
        y = paddle.to_tensor([[1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0]])
                                
        z1 = paddle.cross(x, y)
        # [[-1. -1. -1.]
        #  [ 2.  2.  2.]
        #  [-1. -1. -1.]]

        z2 = paddle.cross(x, y, axis=1)
        # [[0. 0. 0.]
        #  [0. 0. 0.]
        #  [0. 0. 0.]]


