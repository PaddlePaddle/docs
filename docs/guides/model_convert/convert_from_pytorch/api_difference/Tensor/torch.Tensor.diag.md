## [ 仅参数名不一致 ]torch.Tensor.diag

### [torch.Tensor.diag](https://pytorch.org/docs/stable/generated/torch.Tensor.diag.html?highlight=diag#torch.Tensor.diag)

```python
torch.Tensor.diag(diagonal=0)
```

### [paddle.Tensor.diag](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diag_cn.html#diag)

```python
paddle.diag(x, offset=0, padding_value=0, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle  | 备注                                                                                   |
| -------- | ------------- | -------------------------------------------------------------------------------------- |
| -        | x             | 输入的 Tensor，paddle 参数更多。                                                       |
| diagonal | offset        | 考虑的对角线：正值表示上对角线，0 表示主对角线，负值表示下对角线，仅参数名不一致。       |
| -        | padding_value | 使用此值来填充指定对角线以外的区域，仅在输入为一维 Tensor 时生效，默认值为 0。torch 无此参数，paddle 保持默认即可。 |
