## [ 仅参数名不一致 ]torch.Tensor.vsplit
### [torch.Tensor.vsplit](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.vsplit.html#torch.Tensor.vsplit)
```python
torch.Tensor.vsplit(split_size_or_sections)
```

### [paddle.Tensor.vsplit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#vsplit-num-or-indices-name-none)
```python
paddle.Tensor.vsplit(num_or_indices, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| split_size_or_sections           | num_or_indices         | 表示分割的数量或索引，仅参数名不一致。                          |
