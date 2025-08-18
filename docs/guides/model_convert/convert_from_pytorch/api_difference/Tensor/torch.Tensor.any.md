## [ 仅参数名不一致 ]torch.Tensor.any

### [torch.Tensor.any](https://pytorch.org/docs/stable/generated/torch.Tensor.any.html?highlight=torch+tensor+any#torch.Tensor.any)

```python
torch.Tensor.any(dim=None, keepdim=False)
```

### [paddle.Tensor.any](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#any-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.any(axis=None,
                  keepdim=False,
                  name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim    |  axis     | 表示运算的维度，仅参数名不一致。        |
| keepdim    |  keepdim  | 是否在输出 Tensor 中保留减小的维度，参数完全一致。  |
