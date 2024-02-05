## [ 仅参数名不一致 ] torch.Tensor.nansum
### [torch.Tensor.nansum](https://pytorch.org/docs/stable/generated/torch.Tensor.nansum.html?highlight=nansum#torch.Tensor.nansum)

```python
torch.Tensor.nansum(dim=None,
            keepdim=False,
            dtype=None)
```

### [paddle.Tensor.nansum]()

```python
paddle.Tensor.nansum(axis=None,
            keepdim=False,
            dtype=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim           | axis         | 需要求和的维度，仅参数名不一致。                          |
| keepdim       | keepdim      | 结果是否需要保持维度不变，仅参数名不一致。                 |
| dtype         | dtype        | 返回的 Tensor 的类型，仅参数名不一致。                    |
