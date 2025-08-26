## [ 仅参数名不一致 ]torch.Tensor.all

### [torch.Tensor.all](https://pytorch.org/docs/stable/generated/torch.Tensor.all.html?highlight=torch+tensor+all#torch.Tensor.all)

```python
torch.Tensor.all(dim=None, keepdim=False)
```

### [paddle.Tensor.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#all-axis-none-keepdim-false-name-none)

```python
paddle.Tensor.all(axis=None,
                  keepdim=False,
                  name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim    |  axis     | 表示运算的维度，仅参数名不一致。        |
| keepdim    |  keepdim  | 是否在输出 Tensor 中保留减小的维度，参数完全一致。  |
