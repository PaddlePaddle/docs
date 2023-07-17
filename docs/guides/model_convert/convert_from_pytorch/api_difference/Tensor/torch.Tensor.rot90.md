## [ 仅参数名不一致 ]torch.Tensor.rot90

### [torch.Tensor.rot90](https://pytorch.org/docs/stable/generated/torch.Tensor.rot90.html?highlight=torch+tensor+rot90#torch.Tensor.rot90)

```python
torch.Tensor.rot90(k, dims)
```

### [paddle.rot90](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/rot90_cn.html)

```python
paddle.rot90(x, k=1, axes=[0, 1], name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| k           | k         | 旋转方向和次数。 |
| dims    | axes    | 指定旋转的平面，仅参数名不一致。     |
