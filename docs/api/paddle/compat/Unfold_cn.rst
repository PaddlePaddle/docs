.. _cn_api_paddle_compat_Unfold:

Unfold
-------------------------------

.. py:class:: paddle.compat.Unfold(kernel_size, dilation=1, padding=0, stride=1)


PyTorch 兼容的 :ref:`cn_api_paddle_nn_Unfold` 版本：
    - 关键字参数使用单数形式（例如： ``kernel_size``  而非 kernel_sizes）
    -  ``padding``  仅支持输入长度为 1（整数）或 2 的列表，禁止使用 Size4 格式。如需更灵活的输入版本，请使用 :ref:`cn_api_paddle_nn_Unfold`
    - 所有输入参数支持  ``Tensor``  或  ``pir.Value``  类型（将自动转换为列表）
    - 其他特性与 :ref:`cn_api_paddle_nn_Unfold` 完全一致

使用前请详细参考：`【仅参数名不一致】torch.nn.Unfold`_ 以确定是否使用此模块。

.. _【仅参数名不一致】torch.nn.Unfold: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/nn/torch.nn.Unfold.html

**样例**：

::

      Given:
        x.shape = [5, 10, 25, 25]
        kernel_size = [3, 3]
        stride = 1
        padding = 1

      Return:
        out.shape = [5, 90, 625]

功能说明
:::::::::::
实现卷积中的 im2col 操作，将卷积核覆盖区域的元素重排列为列向量。输入形状为 [N, C, H, W] 的  ``x``  将输出形状为 [N, Cout, Lout] 的张量。


参数
:::::::::::

        - **kernel_size** (int|list|tuple|Tensor) - 卷积核尺寸。接受  ``[k_h, k_w]``  或整数  ``k``  （视为   ``[k, k]`` ）
        - **dilation** (int|list|tuple|Tensor, 可选) - 卷积膨胀系数。接受单整数或  ``[dilation_h, dilation_w]`` 。单整数  ``dilation``  视为  ``[dilation, dilation]`` 。默认为 1
        - **padding** (int|list|tuple|Tensor, 可选) - 各维度填充大小。接受单整数或  ``[padding_h, padding_w]`` 。 ``[padding_h, padding_w]``  将扩展为  ``[padding_h, padding_w, padding_h, padding_w]`` 。单整数  ``padding``  将转为  ``[padding, padding, padding, padding]`` 。默认为 0
        - **stride** (int|list|tuple|Tensor, 可选) - 滑动步长。接受单整数或  ``[stride_h, stride_w]`` 。单整数  ``stride``  视为  ``[stride, stride]`` 。默认为 1

形状
:::::::::
 - **输入** : 4-D Tensor，形状为[N, C, H, W]，数据类型为 float32 或者 float64
 - **输出**：形状如上面所描述的[N, Cout, Lout]，Cout 每一个滑动 block 里面覆盖的元素个数，Lout 是滑动 block 的个数，数据类型与  ``x``  相同


代码示例
::::::::::::

COPY-FROM: paddle.compat.Unfold
