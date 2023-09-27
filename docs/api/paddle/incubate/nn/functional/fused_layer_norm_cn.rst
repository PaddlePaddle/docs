.. _cn_api_paddle_incubate_nn_functional_fused_layer_norm:

fused_layer_norm
-------------------------------

.. py:function:: paddle.incubate.nn.functional.fused_layer_norm(x, norm_weight, norm_bias, epsilon, residual_alpha=1.0, begin_norm_axis=1, bias=None, residual=None, quant_scale=- 1, quant_round_type=0, quant_max_bound=0, quant_min_bound=0)

应用 Fused LayerNorm 内核，提供了更高的 GPU 利用率。同时，支持模式融合 LayerNorm(bias + residual_alpha * residual + x)。

当 norm_weight 和 norm_bias 为 None 时，返回融合的 (bias + residual_alpha * residual + x)。

fused_layer_norm 算子目前只支持在 GPU 下运行。

参数
::::::::::::
    - **x** (Tensor) - 输入 ``Tensor``。
    - **norm_weight** (Tensor) - 用于仿射输出的权重张量。
    - **norm_bias** (Tensor) - 用于仿射输出的偏置张量。
    - **epsilon** (float) - 一个小的浮点数，用于避免除以零。
    - **residual_alpha** (float) - 用于残差的缩放因子，默认为 1。
    - **begin_norm_axis** (int) - 归一化的起始轴，默认为 1。
    - **bias** (可选|Tensor) - 前一层的偏置。
    - **residual** (可选|Tensor) - 输入的残差。
    - **quant_scale** (float) - 量化缩放因子。
    - **quant_round_type** (float) - 量化四舍五入类型。
    - **quant_max_bound** (float) - 量化裁剪的最大边界值。
    - **quant_min_bound** (float) - 量化裁剪的最小边界值。


返回
::::::::::::
输出``Tensor``

形状
::::::::::::
``Tensor``，形状同 ``x`` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.incubate.nn.functional.fused_layer_norm
