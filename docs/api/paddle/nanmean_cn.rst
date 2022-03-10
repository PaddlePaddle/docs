.. _cn_api_tensor_cn_nanmean:

nanmean
-------------------------------

.. py:function:: paddle.nanmean(x, axis=None, keepdim=False, name=None)



该OP沿 ``axis`` 计算 ``x`` 的平均值,且忽略掉 ``NaNs`` 值。

参数
::::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：float32、float64。
    - axis (int|list|tuple, 可选) - 指定对 ``x`` 进行计算的轴。``axis`` 可以是int、list(int)、tuple(int)。如果 ``axis`` 包含多个维度，则沿着 ``axis`` 中的所有轴进行计算。``axis`` 或者其中的元素值应该在范围[-D, D)内，D是 ``x`` 的维度。如果 ``axis`` 或者其中的元素值小于0，则等价于 :math:`axis + D` 。如果 ``axis`` 是None，则对 ``x`` 的全部元素计算平均值。默认值为None。
    - keepdim (bool, 可选) - 是否在输出Tensor中保留减小的维度。如果 ``keepdim`` 为True，则输出Tensor和 ``x`` 具有相同的维度(减少的维度除外，减少的维度的大小为1)。否则，输出Tensor的形状会在 ``axis`` 上进行squeeze操作。默认值为False。
    - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，沿着 ``axis`` 进行平均值计算的结果且忽略掉 ``NaNs`` 值，数据类型和 ``x`` 相同。

代码示例
::::::::::

.. code-block:: python

    import paddle

    # x is a Tensor with following elements:
    #    [[nan, 0.3, 0.5, 0.9]
    #     [0.1, 0.2, -nan, 0.7]]
    # Each example is followed by the corresponding output tensor.
    x = np.array([[float('nan'), 0.3, 0.5, 0.9],
                    [0.1, 0.2, float('-nan'), 0.7]]).astype(np.float32)
    x = paddle.to_tensor(x)

    out1 = nanmean(x)                       #[0.45000002]                   
    out2 = nanmean(x,axis=0)                #[0.1        0.25       0.5        0.79999995]
    out3 = nanmean(x,axis=0,keepdim=True)   #[[0.1        0.25       0.5        0.79999995]]
    out4 = nanmean(x,axis=1)                #[0.56666666 0.33333334]
    out5 = nanmean(x,axis=1,keepdim=True)   #[[0.56666666]
                                            # [0.33333334]]
    
    # y is a Tensor with shape [2, 2, 2] and elements as below:
    #      [[[1, nan], [3, 4]],
    #      [[5, 6], [-nan, 8]]]
    # Each example is followed by the corresponding output tensor.
    y = np.array([[[1, float('nan')], [3, 4]], 
                    [[5, 6], [float('-nan'), 8]]])
    y = paddle.to_tensor(y)
    out5 = paddle.nanmean(y, axis=[1, 2]) # [3. 6.]
    out6 = paddle.nanmean(y, axis=[0, 1]) # [2.66666667 6.33333333]
