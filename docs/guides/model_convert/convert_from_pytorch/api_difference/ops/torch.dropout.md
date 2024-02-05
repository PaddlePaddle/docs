## [ 仅参数名不一致 ]torch.dropout

### [torch.dropout](https://pytorch.org/docs/stable/jit_builtin_functions.html#supported-pytorch-functions)

```python
torch.dropout(input: Tensor,
              p: float,
              train: bool)
```

### [paddle.nn.functional.dropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/dropout_cn.html#dropout)

```python
paddle.nn.functional.dropout(x, p=0.5, axis=None, training=True, mode='upscale_in_train', name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入的多维 Tensor，数据类型为：float16、float32、float64。 |
| p       | p            | 将输入节点置 0 的概率，即丢弃概率。默认值为 0.5。 |
| train   | training     | 标记是否为训练阶段。默认值为 True。 |
