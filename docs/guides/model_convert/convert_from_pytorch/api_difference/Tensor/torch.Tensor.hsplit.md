## [ 仅参数名不一致 ]torch.Tensor.hsplit

### [torch.Tensor.hsplit](https://pytorch.org/docs/stable/generated/torch.Tensor.hsplit.html)

```python
torch.Tensor.hsplit(split_size_or_sections)
```

### [paddle.Tensor.hsplit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#hsplit-num_or_indices-name-none)

```python
paddle.Tensor.hsplit(num_or_indices, name=None)
```

其中 Paddle 相比 Pytorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| split_size_or_sections           | num_or_indices         | 表示分割的数量或索引，仅参数名不一致。                          |
