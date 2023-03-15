## [ paddle 参数更多 ]torch.Tensor.diag

### [torch.diag](https://pytorch.org/docs/stable/generated/torch.diag.html?highlight=diag#torch.diag)

```python
torch.diag(input, diagonal=0, *, out=None)
```

### [paddle.diag](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/diag_cn.html#diag)

```python
paddle.diag(x, offset=0, padding_value=0, name=None)
```

两者功能一致且参数用法一致，paddle 参数更多，具体如下：

### 参数映射

| PyTorch                     | PaddlePaddle                     | 备注                                                                                   |
| --------------------------- | -------------------------------- | -------------------------------------------------------------------------------------- |
| <center> input </center>    | <center> x </center>             | 输入的 Tensor，仅参数名不同。                                                          |
| <center> diagonal </center> | <center> offset </center>        | 考虑的对角线：正值表示上对角线，0 表示主对角线，负值表示下对角线，仅参数名不同。       |
| <center> - </center>        | <center> padding_value </center> | paddle：使用此值来填充指定对角线以外的区域。仅在输入为一维 Tensor 时生效。默认值为 0。 |
