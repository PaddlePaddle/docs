## [ 仅参数名不一致 ]torch.any

### [torch.any](https://pytorch.org/docs/stable/generated/torch.any.html?highlight=any#torch.any)

```python
torch.any(input, dim=None, keepdim=False, *, out=None)
```

### [paddle.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/any_cn.html#any)

```python
paddle.any(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x           | 输入的多维 Tensor ，参数完全一致。                   |
| dim    |  axis     | 表示运算的维度，仅参数名不一致。        |
| keepdim    |  keepdim  | 是否在输出 Tensor 中保留减小的维度，参数完全一致。  |
