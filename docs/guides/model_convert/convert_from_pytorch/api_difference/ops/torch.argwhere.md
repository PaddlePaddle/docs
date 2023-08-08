## [ 仅 paddle 参数更多 ]torch.argwhere
### [torch.argwhere](https://pytorch.org/docs/stable/generated/torch.argwhere.html#torch.argwhere)

```python
torch.argwhere(input)
```

### [paddle.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nonzero_cn.html#nonzero)

```python
paddle.nonzero(x, as_tuple=False)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入的 Tensor ，仅参数名不一致。                   |
| - | <font color='red'> as_tuple </font>   | 返回格式。是否以 1-D Tensor 构成的元组格式返回。 Pytorch 无此参数， Paddle 保持默认即可。  |
