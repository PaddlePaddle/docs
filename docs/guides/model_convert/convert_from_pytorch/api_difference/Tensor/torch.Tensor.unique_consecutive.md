## [仅 paddle 参数更多]torch.Tensor.unique_consecutive

### [torch.Tensor.unique_consecutive](https://pytorch.org/docs/1.13/generated/torch.Tensor.unique_consecutive.html#torch.Tensor.unique_consecutive)

```python
torch.Tensor.unique_consecutive(return_inverse=False, return_counts=False, dim=None)
```

### [paddle.Tensor.unique_consecutive]()

```python
paddle.Tensor.unique_consecutive(return_inverse=False, return_counts=False, axis=None, dtype='int64', name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

|    PyTorch     |  PaddlePaddle  |                             备注                             |
| :------------: | :------------: | :----------------------------------------------------------: |
| return_inverse | return_inverse | 表示输入 Tensor 的元素对应在连续不重复元素中的索引。参数完全一致。 |
| return_counts  | return_counts  | 表示每个连续不重复元素在输入 Tensor 中的个数。参数完全一致。 |
|      dim       |      axis      |              表示进行运算的轴，仅参数名不一致。              |
|       -        |     dtype      | 表示设置 inverse 或 counts 的类型。PyTorch 无此参数，Paddle 保持默认即可。 |
