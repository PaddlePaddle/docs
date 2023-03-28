## [ 仅参数名不一致 ] torch.Tensor.chunk

### [torch.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk)

```python
torch.chunk(input, chunks, dim=0)
```

### [paddle.chunk](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/chunk_cn.html#chunk)

```python
paddle.chunk(x, chunks, axis=0, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch    | PaddlePaddle | 备注                                                   |
|------------| ------------ | ------------------------------------------------------ |
| input      | x           | 表示输入的 Tensor，仅参数名不一致。               |
| chunks     | chunks           |  表示将输入 Tensor 划分成多少个相同大小的子 Tensor。               |
| dim        | axis           |   表示需要分割的维度，仅参数名不一致。               |
