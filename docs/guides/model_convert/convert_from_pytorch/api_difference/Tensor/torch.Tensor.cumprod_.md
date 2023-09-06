## [ 参数完全一致 ]torch.Tensor.cumprod_

### [torch.Tensor.cumprod_](https://pytorch.org/docs/stable/generated/torch.Tensor.cumprod_.html)

```python
torch.Tensor.cumprod_(dim, dtype=None)
```

### [paddle.Tensor.cumprod_]()

```python
paddle.Tensor.cumprod_(dim=None, dtype=None)
```

两者功能一致且参数用法一致， 具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                                                                 |
| ------- | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| dim     | dim          | 指明需要累乘的维度。                                                                                                 |
| dtype   | dtype        | 返回张量所需的数据类型。dtype 如果指定，则在执行操作之前将输入张量转换为指定的 dtype，这对于防止数据类型溢出很有用。 |
