## [仅参数名不一致]torch.histogramdd

### [torch.histogramdd](https://pytorch.org/docs/stable/generated/torch.histogramdd.html#torch-histogramdd)

```python
torch.histogramdd(input, bins, *, range=None, weight=None, density=False)
```

### [paddle.histogramdd](https://github.com/PaddlePaddle/Paddle/blob/a19227d9ee0e351363a4bb27b50b1becbec58a6c/python/paddle/tensor/linalg.py#L3875)

```python
paddle.histogramdd(x, bins=10, ranges=None, density=False, weights=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                          |
| ------- | ------------ | ----------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。 |
| bins    | bins         | 直方图 bins(直条)的个数序列。 |
| range   | ranges       | bins 的范围，仅参数名不一致。 |
| weight  | weights      | 权重，仅参数名不一致。        |
| density | density      | 结果中每个 bin 是否包含权重。 |
