### [ paddle 参数更多 ]torch.nn.functional.channel_shuffle

### [torch.nn.functional.channel_shuffle](https://pytorch.org/docs/stable/generated/torch.nn.ChannelShuffle.html)

```python
torch.nn.functional.channel_shuffle(input,
                                    groups)
```

### [paddle.nn.functional.channel_shuffle](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/channel_shuffle_cn.html#channel-shuffle)

```python
paddle.nn.functional.channel_shuffle(x,
                                    groups,
                                    data_format='NCHW',
                                    name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。  |
| groups         | groups            | 表示要把通道分成的组数。  |
| -             | data_format            | 数据格式，可选：NCHW 或 NHWC |
