## [ 仅参数名不一致 ]torch.combinations

### [torch.combinations](https://pytorch.org/docs/stable/generated/torch.combinations.html#torch.combinations)

```python
torch.combinations(input, r=2, with_replacement=False)
```

### [paddle.combinations](https://github.com/PaddlePaddle/Paddle/blob/8932f1c5e26788ab1eed226e70fafb1ea67ce737/python/paddle/tensor/math.py#L7099)

```python
paddle.combinations(x, r=2, with_replacement=False, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch          | PaddlePaddle     | 备注                          |
| ---------------- | ---------------- | ----------------------------- |
| input            | x                | 输入 Tensor，仅参数名不一致。 |
| r                | r                | 需要合并元素数量。            |
| with_replacement | with_replacement | 是否允许合并中替换。          |
