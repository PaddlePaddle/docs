## [ 仅参数名不一致 ]torch.Tensor.as_strided
### [torch.Tensor.as_strided](https://pytorch.org/docs/stable/generated/torch.Tensor.as_strided.html?highlight=as_strided#torch.Tensor.as_strided)

```python
torch.Tensor.as_strided(size,
                stride,
                storage_offset=None)
```

### [paddle.Tensor.as_strided](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#as-strided-x-shape-stride-offset-0-name-none)

```python
paddle.Tensor.as_strided(shape,
                stride,
                offset=0,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size           | shape            | 表示输出 Tensor 的维度, 仅参数名不一致。               |
| stride           | stride            | 表示输出 Tensor 的 stride。               |
| storage_offset   | offset            | 表示偏移量, 仅参数名不一致。    |
