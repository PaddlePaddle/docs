## [参数完全一致]torch.Tensor.nan_to_num_

### [torch.Tensor.nan_to_num_](https://pytorch.org/docs/stable/generated/torch.Tensor.nan_to_num_.html#torch.Tensor.nan_to_num_)

```python
torch.Tensor.nan_to_num_(nan=0.0, posinf=None, neginf=None)
```

### [paddle.Tensor.nan_to_num_]()

```python
paddle.Tensor.nan_to_num_(nan=0.0, posinf=None, neginf=None)
```

两者功能一致，且参数用法一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注            |
| ------- | ------------ | --------------- |
| nan     | nan          | NaN 的替换值。  |
| posinf  | posinf       | +inf 的替换值。 |
| neginf  | neginf       | -inf 的替换值。 |
