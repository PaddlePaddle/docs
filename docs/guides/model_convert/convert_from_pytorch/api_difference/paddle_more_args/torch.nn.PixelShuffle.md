## [ paddle 参数更多 ]torch.nn.PixelShuffle
### [torch.nn.PixelShuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html?highlight=pixel#torch.nn.PixelShuffle)

```python
torch.nn.PixelShuffle(upscale_factor)
```

### [paddle.nn.PixelShuffle](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/PixelShuffle_cn.html)

```python
paddle.nn.PixelShuffle(upscale_factor,
                        data_format='NCHW',
                        name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| upscale_factor   | upscale_factor | 表示增大空间分辨率的增大因子。                   |
| -   | data_format | 指定输入的 format ， PyTorch 无此参数， Paddle 保持默认即可。                  |
