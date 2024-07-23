## [ paddle 参数更多 ]torch.nn.PixelUnshuffle
### [torch.nn.PixelUnshuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html?highlight=pixel#torch.nn.PixelUnshuffle)

```python
torch.nn.PixelUnshuffle(downscale_factor)
```

### [paddle.nn.PixelUnshuffle](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/PixelUnshuffle_cn.html)

```python
paddle.nn.PixelUnshuffle(downscale_factor,
                        data_format='NCHW',
                        name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| downscale_factor   | downscale_factor | 表示减小空间分辨率的减小因子。                   |
| -   | data_format | 指定输入的 format ， PyTorch 无此参数， Paddle 保持默认即可。                  |
