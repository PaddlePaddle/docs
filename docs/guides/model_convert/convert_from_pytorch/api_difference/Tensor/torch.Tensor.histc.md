## [ 参数完全一致 ]torch.Tensor.histc

### [torch.Tensor.histc](https://pytorch.org/docs/1.13/generated/torch.Tensor.histc.html?highlight=torch+tensor+histc#torch.Tensor.histc)

```python
torch.Tensor.histc(bins=100, min=0, max=0)
```

### [paddle.Tensor.histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#histogram-bins-100-min-0-max-0)

```python
paddle.Tensor.histogram(bins=100, min=0, max=0, name=None)
```

两者功能一致且参数完全一致。

### 参数映射

| PyTorch                           | PaddlePaddle                 | 备注                                                   |
|-----------------------------------|------------------------------| ------------------------------------------------------ |
| <font color='red'> bins </font> | <font color='red'> bins </font> | 直方图 bins(直条)的个数，默认为 100。                                     |
| <font color='red'> min </font> | <font color='red'> min </font> | range 的下边界(包含)，默认为 0。                                     |
| <font color='red'> max </font> | <font color='red'> max </font> | range 的上边界(包含)，默认为 0。                                     |
