## [ 仅 paddle 参数更多 ]torch.Tensor.argwhere
### [torch.Tensor.argwhere](https://pytorch.org/docs/stable/generated/torch.Tensor.argwhere.html#torch.Tensor.argwhere)

```python
torch.Tensor.argwhere()
```

### [paddle.Tensor.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#nonzero-as-tuple-false)

```python
paddle.Tensor.nonzero(as_tuple=False)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| - | <font color='red'> as_tuple </font>   | 返回格式。是否以 1-D Tensor 构成的元组格式返回。 PyTorch 无此参数， Paddle 保持默认即可。  |
