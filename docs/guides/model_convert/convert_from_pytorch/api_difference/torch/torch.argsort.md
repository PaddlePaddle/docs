## [ 仅参数名不一致 ]torch.argsort
### [torch.argsort](https://pytorch.org/docs/stable/generated/torch.argsort.html#torch.argsort)

```python
torch.argsort(input, dim=-1, descending=False, stable=False)
```

### [paddle.argsort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/argsort_cn.html#argsort)

```python
paddle.argsort(x, axis=-1, descending=False, stable=False, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入的多维 Tensor ，仅参数名不一致。                   |
| <font color='red'> dim </font> | <font color='red'> axis </font>    | 指定进行运算的轴，仅参数名不一致。  |
| descending |  descending | 是否使用降序排列。  |
|  stable  |  stable   | 是否使用稳定排序。  |
