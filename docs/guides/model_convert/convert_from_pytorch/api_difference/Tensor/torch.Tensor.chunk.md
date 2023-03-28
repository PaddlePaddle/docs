## [ 仅参数名不一致 ] torch.Tensor.chunk

### [torch.Tensor.chunk](https://pytorch.org/docs/stable/generated/torch.Tensor.chunk.html?highlight=chunk#torch.Tensor.chunk)

```python
torch.Tensor.chunk(chunks, dim=0)
```

### [paddle.Tensor.chunk](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#cosh-name-none)

```python
paddle.Tensor.chunk(chunks, axis=0, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch    | PaddlePaddle | 备注                                                   |
|------------| ------------ | ------------------------------------------------------ |
| chunks     | chunks         |  表示将输入 Tensor 划分成多少个相同大小的子 Tensor。               |
| dim        | axis           |   表示需要分割的维度，仅参数名不一致。               |
