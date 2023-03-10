## torch.nn.functional.one_hot

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

两者功能一致，仅参数名不一致，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor 。                                     |
| pad          | pad         | 表示一个 one-hot 向量的长度 。                                     |
| mode          | mode         | 表示填充的模式。                                     |
| value          | value         | 表示填充的值，mode为'constant'时有效                           |
| -        | data_format |  指定输入的数据格式 |

在实际使用过程中，`data_format` 参数需要根据输入参数进行指定