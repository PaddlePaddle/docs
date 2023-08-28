## [参数完全一致]torch.einsum

### [torch.einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html#torch.einsum)

```python
torch.einsum(equation, *operands)
```

### [paddle.einsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/einsum_cn.html)

```python
paddle.einsum(equation, *operands)
```

其中功能一致, 参数完全一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注       |
| -------- | ------------ | ---------- |
| equation | equation     | 求和标记。 |
| operands | operands     | 输入张量。 |
