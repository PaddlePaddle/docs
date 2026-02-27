## [ 仅参数名不一致 ]torch.Tensor.dsplit
api 存在重载情况，分别如下：

-------------------------------------------------------------------------------------------------

### [torch.Tensor.dsplit](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dsplit.html#torch.Tensor.dsplit)
```python
torch.Tensor.dsplit(sections)
```

### [paddle.Tensor.dsplit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#dsplit-num-or-indices-name-none)
```python
paddle.Tensor.dsplit(num_or_indices, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| sections           | num_or_indices         | 表示分割的数量，仅参数名不一致。                          |

-------------------------------------------------------------------------------------------------

### [torch.Tensor.dsplit](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.dsplit.html#torch.Tensor.dsplit)
```python
torch.Tensor.dsplit(indices)
```

### [paddle.Tensor.dsplit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#dsplit-num-or-indices-name-none)
```python
paddle.Tensor.dsplit(num_or_indices, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| indices           | num_or_indices         | 表示分割的索引，仅参数名不一致。                          |
