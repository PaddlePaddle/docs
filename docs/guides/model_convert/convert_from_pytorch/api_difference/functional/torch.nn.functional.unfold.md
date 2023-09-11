## [参数名不一致]torch.nn.functional.unfold

### [torch.nn.functional.unfold](https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html#torch.nn.functional.unfold)

```python
torch.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)
```

### [paddle.nn.functional.unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/unfold_cn.html)

```python
paddle.nn.functional.unfold(x, kernel_size, strides=1, paddings=0, dilation=1, name=None)
```

其中功能一致, 仅参数名不一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                              |
| ----------- | ------------ | --------------------------------- |
| input       | x            | 输入 4-D Tensor，仅参数名不一致。 |
| kernel_size | kernel_size  | 卷积核的尺寸。                    |
| dilation    | dilation     | 卷积膨胀。                        |
| padding     | paddings     | 每个维度的扩展，仅参数名不一致。  |
| stride      | strides      | 卷积步长，仅参数名不一致。        |
