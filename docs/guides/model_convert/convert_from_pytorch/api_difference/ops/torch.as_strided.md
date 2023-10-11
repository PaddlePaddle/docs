## [ 仅参数名不一致 ]torch.as_strided
### [torch.as_strided](https://pytorch.org/docs/stable/generated/torch.as_strided.html?highlight=as_strided#torch.as_strided)

```python
torch.as_strided(input,
                size,
                stride,
                storage_offset=None)
```

### [paddle.as_strided](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/as_strided_cn.html#as-strided)

```python
paddle.as_strided(x,
                shape,
                stride,
                offset=0,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor 。                                     |
| size           | shape            | 表示输出 Tensor 的维度, 仅参数名不一致。               |
| stride           | stride            | 表示输出 Tensor 的 stride。               |
| storage_offset   | offset            | 表示偏移量    |
