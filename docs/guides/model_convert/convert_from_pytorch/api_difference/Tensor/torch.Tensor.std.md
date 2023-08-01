## [ 仅参数名不一致 ]torch.Tensor.std

### [torch.Tensor.std](https://pytorch.org/docs/stable/generated/torch.Tensor.std.html?highlight=torch+tensor+std#torch.Tensor.std)

```python
torch.Tensor.std(dim=None, *, correction=1, keepdim=False)
```

### [paddle.Tensor.std](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#std-axis-none-unbiased-true-keepdim-false-name-none)

```python
paddle.Tensor.std(axis=None, unbiased=True, keepdim=False, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | ------- |
| dim      | axis        | 指定对 Tensor 进行计算的轴，仅参数名不一致。 |
| correction | unbiased   | 是否使用无偏估计来计算标准差。PyTorch 支持 int 类型，Paddle 支持 bool/int 类型。仅参数名不一致。|
| keepdim   | keepdim   | 是否在输出 Tensor 中保留减小的维度。 |
