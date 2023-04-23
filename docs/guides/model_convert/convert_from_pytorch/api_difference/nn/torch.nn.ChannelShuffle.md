## [ 仅 paddle 参数更多 ]torch.nn.ChannelShuffle

### [torch.nn.ChannelShuffle](https://pytorch.org/docs/stable/generated/torch.nn.ChannelShuffle.html?highlight=channelshuffle#torch.nn.ChannelShuffle)

```python
torch.nn.ChannelShuffle(groups)
```

### [paddle.nn.ChannelShuffle](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ChannelShuffle_cn.html)

```python
paddle.nn.ChannelShuffle(groups, data_format='NCHW', name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| groups          | groups         | 表示要把通道分成的组数 。                                     |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
