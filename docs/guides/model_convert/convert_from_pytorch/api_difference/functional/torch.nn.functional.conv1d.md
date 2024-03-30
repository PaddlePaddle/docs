## [ 仅参数名不一致 ]torch.nn.functional.conv1d

### [torch.nn.functional.conv1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html?highlight=conv1d#torch.nn.functional.conv1d)

```python
torch.nn.functional.conv1d(input,
                           weight,
                           bias=None,
                           stride=1,
                           padding=0,
                           dilation=1,
                           groups=1)
```

### [paddle.nn.functional.conv1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/conv1d_cn.html)

```python
paddle.nn.functional.conv1d(x,
                            weight,
                            bias=None,
                            stride=1,
                            padding=0,
                            dilation=1,
                            groups=1,
                            data_format='NCL',
                            name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| weight          | weight         | 表示权重 Tensor 。                                     |
| bias          | bias         | 表示偏置项 。                                     |
| stride          | stride         | 表示步长 。                                     |
| padding          | padding         | 表示填充大小 。                                     |
| dilation          | dilation         | 表示空洞大小 。                                     |
| groups          | groups         | 表示分组数 。                                     |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
