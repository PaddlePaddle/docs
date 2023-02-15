## torch.chunk
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
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入变量，数据类型为 bool, float16, float32，float64，int32，int64 的多维 Tensor。   |
| dim          | axis         | 表示需要分割的维度。 |
