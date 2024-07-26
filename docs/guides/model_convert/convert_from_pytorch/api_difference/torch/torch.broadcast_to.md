## [ 仅参数名不一致 ]torch.broadcast_to

### [torch.broadcast_to](https://pytorch.org/docs/stable/generated/torch.broadcast_to.html?highlight=broadcast_to#torch.broadcast_to)

```python
torch.broadcast_to(input,
                   size)
```

### [paddle.broadcast_to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/broadcast_to_cn.html#broadcast-to)

```python
paddle.broadcast_to(x,
                    shape,
                    name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>| <font color='red'>x</font> | 表示输入的 Tensor ，仅参数名不一致。  |
| size | shape | 表示扩展后 Tensor 的 shape，仅参数名不一致。  |
