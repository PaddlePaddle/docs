## [仅参数名不一致]torch.nn.functional.pdist

### [torch.nn.functional.pdist](https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html#torch.nn.functional.pdist)

```python
torch.nn.functional.pdist(input, p=2)
```

### [paddle.nn.functional.pdist](https://github.com/PaddlePaddle/Paddle/blob/210442ec30e5038809865a6105dd38308d1df2e0/python/paddle/nn/functional/distance.py#L111)

```python
paddle.nn.functional.pdist(x, p=2.0)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                            |
| ------- | ------------ | ------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。 |
| p       | p            | 计算 p-norm 距离的 p 参数。     |
