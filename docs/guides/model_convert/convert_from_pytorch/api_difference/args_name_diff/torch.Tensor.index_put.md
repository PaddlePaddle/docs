## [ 仅参数名不一致 ]torch.Tensor.index_put
### [torch.Tensor.index\_put](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_put.html#torch.Tensor.index_put)
```python
torch.Tensor.index_put(indices, values, accumulate=False)
```

### [paddle.Tensor.index\_put](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#index-put-indices-value-accumulate-false-name-none)
```python
paddle.Tensor.index_put(indices, value, accumulate=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| indices | indices | 包含用来索引的 tensors 的元组。数据类型为 int32，int64，bool。 |
| values | value | 用来给 x 赋值的 Tensor，仅参数名不一致。  |
| accumulate | accumulate | 指定是否将 value 加到 x 的参数。 默认值为 False。 |
