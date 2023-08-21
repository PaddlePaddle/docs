## [仅参数名不一致]torch.Tensor.unsqueeze_

### [torch.Tensor.unsqueeze_](https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze_.html#torch-tensor-unsqueeze)

```python
torch.Tensor.unsqueeze_(dim)
```

### [paddle.Tensor.unsqueeze_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#id22)

```python
paddle.Tensor.unsqueeze_(axis, name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                备注                |
| :-----: | :----------: | :--------------------------------: |
|   dim   |     axis     | 表示进行运算的轴，仅参数名不一致。 |
