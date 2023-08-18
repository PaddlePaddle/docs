## [仅参数名不一致]torch.kron

### [torch.kron](https://pytorch.org/docs/stable/generated/torch.kron.html#torch-kron)

```python
torch.kron(input, other, *, out=None)
```

### [paddle.kron](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/kron_cn.html#kron)

```python
paddle.kron(x, y, out=None, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                |
| ------- | ------------ | --------------------------------------------------- |
| input   | x            | 表示 Kron OP 输入的第一个 Tensor ，仅参数名不一致。    |
| other   | y            | 表示 Kron OP 输入的第二个 Tensor ，仅参数名不一致。    |
| out     | out          | 表示输出的 Tensor。                                  |
