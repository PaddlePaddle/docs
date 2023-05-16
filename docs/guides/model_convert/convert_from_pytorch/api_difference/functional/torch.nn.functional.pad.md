## [ 仅 paddle 参数更多 ]torch.nn.functional.pad

### [torch.nn.functional.pad](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html)

```python
torch.nn.functional.pad(input,
                            pad,
                            mode='constant',
                            value=None)
```

### [paddle.nn.functional.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/pad_cn.html#pad)

```python
paddle.nn.functional.pad(x,
                            pad,
                            mode='constant',
                            value=0.0,
                            data_format='NCHW',
                            name=None)
```

两者功能一致，其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor，仅参数名不一致。                                     |
| pad          | pad         | 表示一个 one-hot 向量的长度 。                                     |
| mode          | mode         | 表示填充的模式。                                     |
| value          | value         | 表示填充的值，mode 为'constant'时有效 。                |
| -        | data_format |  指定输入的数据格式, PyTorch 无此参数， Paddle 保持默认即可。 |

在实际使用过程中，`data_format` 参数需要根据输入参数进行指定
