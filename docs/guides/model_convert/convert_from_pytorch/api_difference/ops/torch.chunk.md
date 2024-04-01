## [ 仅参数名不一致 ]torch.chunk
### [torch.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html?highlight=chunk#torch.chunk)

```python
torch.chunk(input,
            chunks,
            dim=0)
```

### [paddle.chunk](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/chunk_cn.html#chunk)

```python
paddle.chunk(x,
             chunks,
             axis=0,
             name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> chunks </font> | <font color='red'> chunks </font> | 表示将输入 Tensor 划分成多少个相同大小的子 Tensor。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> |表示需要分割的维度 ，仅参数名不一致。  |
