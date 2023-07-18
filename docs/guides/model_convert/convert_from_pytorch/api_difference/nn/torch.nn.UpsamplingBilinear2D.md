## [ 仅 paddle 参数更多 ]torch.nn.UpsamplingBilinear2d

### [torch.nn.UpsamplingBilinear2d](https://pytorch.org/docs/stable/generated/torch.nn.UpsamplingBilinear2d.html?highlight=upsamplingbilinear2d#torch.nn.UpsamplingBilinear2d)

```python
torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)
```

### [paddle.nn.UpsamplingBilinear2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/UpsamplingBilinear2D_cn.html)

```python
paddle.nn.UpsamplingBilinear2D(size=None,scale_factor=None, data_format='NCHW',name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | size         | 表示输出 Tensor 的 size 。                                     |
| scale_factor           | scale_factor            | 表示输入 Tensor 的高度或宽度的乘数因子。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
