.. _cn_api_paddle_nn_functional_dropout1d:

dropout1d
-------------------------------

.. py:function:: paddle.nn.functional.dropout1d(x, p=0.5, training=True, inplace=False)
根据丢弃概率 ``p``，在训练过程中随机将某些 1D 通道置 0（对一个形状为 ``[N, C, L]`` 的 3D Tensor 或 ``[C, L]`` 的 2D Tensor，1D 通道指的是其中的形状为 ``L`` 的 1 维特征图）。

参数
:::::::::
- **x** (Tensor) - 形状为 ``[N, C, L]`` 的 3D Tensor 或 ``[C, L]`` 的 2D Tensor，数据类型为 float16、float32 或 float64。
- **p** (float，可选) - 将输入通道置 0 的概率，即丢弃概率。默认值为 0.5。
- **training** (bool，可选) - 标记是否为训练阶段。若为 False，则直接返回输入。默认值为 True。
- **inplace** (bool，可选) - 是否原地操作。当前版本暂未实现（行为等同于 False），未来版本将支持。默认值为 False。

返回
:::::::::
与输入 ``x`` 形状相同的 ``Tensor``。

代码示例
:::::::::

.. code-block:: python
    import paddle
    # Case 1: 3D input (batched)
    x = paddle.randn([2, 3, 10])  # [N, C, L]
    y_train = paddle.nn.functional.dropout1d(x, p=0.2)
    y_test = paddle.nn.functional.dropout1d(x, p=0.2, training=False)
    print("Original first channel:", x[0, 0, :])
    print("Train output (may be zeroed):", y_train[0, 0, :])
    print("Test output (always unchanged):", y_test[0, 0, :])
    # Case 2: 2D input (single sample)
    x = paddle.randn([3, 8])  # [C, L]
    y = paddle.nn.functional.dropout1d(x, p=0.5)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Zeroed channels count:", paddle.sum(y == 0).item())
