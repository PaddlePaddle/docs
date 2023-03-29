## [ 仅  paddle 参数更多 ] torch.nn.functional.pixel_unshuffle

### [torch.nn.functional.pixel_unshuffle](https://pytorch.org/docs/stable/generated/torch.nn.functional.pixel_unshuffle.html?highlight=pixel_unshuffle#torch.nn.functional.pixel_unshuffle)

```python
torch.nn.functional.pixel_unshuffle(input, downscale_factor)
```

### [paddle.nn.functional.pixel_unshuffle](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/pixel_unshuffle_cn.html)

```python
paddle.nn.functional.pixel_unshuffle(x, downscale_factor, data_format='NCHW', name=None)
```

两者功能一致，其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor，仅参数名不一致。      |
| downscale_factor           | downscale_factor           |   减小空间分辨率的减小因子。               |
| -           | data_format           |   指定输入张量格式，PyTorch 无此参数，Paddle 保持默认即可。               |
