## [ 一致的参数 ] torch.Tensor.nansum
### [torch.Tensor.nansum](https://pytorch.org/docs/1.13/generated/torch.Tensor.nansum.html?highlight=nansum#torch.Tensor.nansum)

```python
Tensor.nansum(dim=None,
            keepdim=False,
            dtype=None)
```

### [paddle.Tensor.nansum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)未找到文档

```python
paddle.Tensor.nansum(axis=None,
            keepdim=False,
            dtype=None)
```

两者功能一致，返回 Tensor 中元素的和，其中 nan 值记为 0

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim          | axis         | 需要求和的维度                                     |
| keepdim          | keepdim         | 结果是否需要保持维度不变                                     |
| dtype          | dtype         | 返回的 Tensor 的类型                                     |
