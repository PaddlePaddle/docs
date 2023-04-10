## [ 仅参数名不同 ]torch.chunk
### [torch.chunk](https://pytorch.org/docs/stable/generated/torch.chunk.html?highlight=chunk#torch.chunk)

```python
torch.chunk(input,
            chunks,
            dim=0)
```

### [paddle.chunk](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/chunk_cn.html#chunk)

```python
paddle.chunk(x,
             chunks,
             axis=0,
             name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入变量，数据类型为 bool, float16, float32，float64，int32，int64 的多维 Tensor。   |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
