## [仅参数名不一致]torch.softmax

### [torch.softmax](https://pytorch.org/docs/stable/generated/torch.softmax.html?highlight=softmax#torch.softmax)

```python
torch.softmax(input, dim, *, dtype=None)
```

### [paddle.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/softmax_cn.html)

```python
paddle.nn.functional.softmax(x, axis=-1, dtype=None, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，返回类型为 Tensor，数据类型为 dtype 或者和 x 相同，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                      |
| ------- | ------------ | ----------------------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。             |
| dim     | axis         | 表示进行 softmax 的维度，仅参数名不一致。 |
| dtype   | dtype        | 表示需要返回的数据类型。                  |
