## [ 仅参数名不一致 ]torch.Tensor.tensor_split
### [torch.Tensor.tensor_split](https://pytorch.org/docs/stable/generated/torch.Tensor.tensor_split.html)

```python
torch.Tensor.tensor_split(indices_or_sections, dim=0)
```

### [paddle.Tensor.tensor_split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#tensor_split-num_or_indices-axis-0-name-none)

```python
paddle.Tensor.tensor_split(num_or_indices, axis=0, name=None)
```

其中 Paddle 相比 Pytorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| indices_or_sections           | num_or_indices         | 表示分割的数量或索引，仅参数名不一致。                          |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。                          |
