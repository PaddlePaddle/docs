## [仅 paddle 参数更多]flash_attn.ops.rms_norm.rms_norm

### [flash_attn.ops.rms_norm.rms_norm](https://github.com/Dao-AILab/flash-attention/blob/d0787acc16c3667156b51ce5b01bdafc7594ed39/flash_attn/ops/rms_norm.py#L14)

```python
flash_attn.ops.rms_norm.rms_norm(x, weight, epsilon)
```

### [paddle.incubate.nn.functional.fused_rms_norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/incubate/nn/functional/fused_rms_norm_cn.html)

```python
paddle.incubate.nn.functional.fused_rms_norm(x, norm_weight, norm_bias, epsilon, begin_norm_axis, bias=None, residual=None, quant_scale=- 1, quant_round_type=0, quant_max_bound=0, quant_min_bound=0)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle      | 备注                |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| x                 | x                 |  |
| weight            | norm_weight       |  |
| epsilon           | epsilon           |  |
| -                 | norm_bias         |  用于仿射输出的偏置张量 |
| -                 | begin_norm_axis   |  归一化的起始轴 |
| -                 | bias              |  前一层的偏置 |
| -                 | residual          |  输入的残差 |
| -                 | quant_scale       |  量化缩放因子 |
| -                 | quant_round_type  |  量化四舍五入类型 |
| return_attn_probs | quant_max_bound   |  量化裁剪的最大边界值 |
|                   | quant_min_bound   |  量化裁剪的最小边界值 |
