## [ 仅参数名不一致 ]torch.Tensor.argsort

### [torch.Tensor.argsort](https://pytorch.org/docs/stable/generated/torch.Tensor.argsort.html)

```python
torch.Tensor.argsort(dim=-1, descending=False, stable=False)
```

### [paddle.Tensor.argsort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#argsort-axis-1-descending-false-name-none)

```python
paddle.Tensor.argsort(axis=-1, descending=False, stable=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注 |
| ---------- | ------------ | -- |
| dim        | axis         | 指定对输入 Tensor 进行运算的轴，仅参数名不一致。   |
| descending | descending   | 指定算法排序的方向。如果设置为 True，算法按照降序排序。如果设置为 False 或者不设置，按照升序排序。默认值为 False。   |
|  stable  |  stable   | 是否使用稳定排序。  |
