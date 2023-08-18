## [ 仅参数名不一致 ]torch.nn.functional.fold

### [torch.nn.functional.fold](https://pytorch.org/docs/stable/generated/torch.nn.functional.fold.html?highlight=functional+fold#torch.nn.functional.fold)

```python
torch.nn.functional.fold(input,
                         output_size,
                         kernel_size,
                         dilation=1,
                         padding=0,
                         stride=1)
```

### [paddle.nn.functional.fold](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/fold_cn.html)

```python
paddle.nn.functional.fold(x,
                         output_sizes,
                         kernel_sizes,
                         strides=1,
                         paddings=0,
                         dilations=1,
                         name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor 。               |
| output_size           | output_sizes           | 表示输出 Tensor 的尺寸。               |
| kernel_size          | kernel_sizes          | 表示卷积核大小。               |
| dilation           | dilations           | 表示卷积膨胀的大小。               |
| padding          | paddings          | 表示每个维度的填充大小。        |
| stride           | strides           | 表示卷积核步长。               |
