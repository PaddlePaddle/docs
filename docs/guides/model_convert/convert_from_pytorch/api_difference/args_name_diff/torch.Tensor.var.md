## [ 仅参数名不一致 ]torch.Tensor.var
### [torch.Tensor.var](https://pytorch.org/docs/stable/generated/torch.Tensor.var.html#torch.Tensor.var)
```python
torch.Tensor.var(dim, unbiased=True, keepdim=False, *, correction=1)
```

### [paddle.Tensor.var](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#var-axis-none-unbiased-true-keepdim-false-name-none)
```python
paddle.Tensor.var(axis=None, unbiased=True, keepdim=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim      | axis        | 指定对 Tensor 进行计算的轴，仅参数名不一致。 |
| unbiased | unbiased   | torch v1.x 中 correction 对应参数，表示是否为无偏估计，从 v2.0 开始弃用。|
| correction | unbiased   | 是否使用无偏估计来计算方差。PyTorch 支持 int 类型，Paddle 支持 bool/int 类型。仅参数名不一致。|
| keepdim   | keepdim   | 是否在输出 Tensor 中保留减小的维度。 |
