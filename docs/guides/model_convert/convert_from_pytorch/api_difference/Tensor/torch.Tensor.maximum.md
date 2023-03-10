## [仅参数名称不一致]torch.Tensor.maximum

### [torch.Tnsor.maximum](https://pytorch.org/docs/stable/generated/torch.Tensor.maximum.html#torch.Tensor.maximum)

```python
torch.Tensor.maximum(other)
```

### [paddle.Tensor.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#maximum-y-axis-1-name-none)

```python
paddle.Tensor.maximum(y, axis=-1, name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

| PyTorch                          | PaddlePaddle                    | 备注                                 |
|----------------------------------|---------------------------------|------------------------------------|
| <font color='red'> other </font> | <font color='red'> y </font>    | 输⼊ Tensor ，仅名称不同。                  |
| <font color='red'> - </font>     | <font color='red'> axis </font> | 指定进行运算的轴，Pytorch无此参数，Paddle保持默认即可。 |