## [ 仅参数名不一致 ]torch.Tensor.roll
### [torch.Tensor.roll](https://pytorch.org/docs/stable/generated/torch.Tensor.roll.html?highlight=torch+tensor+roll#torch.Tensor.roll)

```python
torch.Tensor.roll(shifts, dims)
```

### [paddle.Tensor.roll](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/roll_cn.html#roll)

```python
paddle.Tensor.roll(shifts, axis=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| shifts         | shifts            | 表示偏移量。                   |
| dims          | axis         | 表示滚动的轴，仅参数名不一致。                          |
