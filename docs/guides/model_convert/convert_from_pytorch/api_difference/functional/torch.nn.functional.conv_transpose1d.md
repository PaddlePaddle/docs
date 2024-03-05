## [仅 paddle 参数更多 ]torch.nn.functional.conv_transpose1d

### [torch.nn.functional.conv_transpose1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv_transpose1d.html?highlight=conv_trans#torch.nn.functional.conv_transpose1d)

```python
torch.nn.functional.conv_transpose1d(input,
                                    weight,
                                    bias=None,
                                    stride=1,
                                    padding=0,
                                    output_padding=0,
                                    groups=1,
                                    dilation=1)
```

### [paddle.nn.functional.conv1d_transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/conv1d_transpose_cn.html)

```python
paddle.nn.functional.conv1d_transpose(x,
                                    weight,
                                    bias=None,
                                    stride=1,
                                    padding=0,
                                    output_padding=0,
                                    groups=1,
                                    dilation=1,
                                    output_size=None,
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
| output_padding          | output_padding         | 表示输出形状上尾部一侧额外添加的大小 。                                     |
| groups          | groups         | 表示分组数 。                                     |
| dilation          | dilation         | 表示空洞大小 。                                     |
| -           | output_size           | 表示输出 Tensor 的尺寸， PyTorch 无此参数， Paddle 保持默认即可。        |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
