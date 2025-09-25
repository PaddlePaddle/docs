## [ 参数完全一致 ] torch.Tensor.nonzero

### [torch.Tensor.nonzero](https://pytorch.org/docs/stable/generated/torch.Tensor.nonzero.html?highlight=nonzero#torch.Tensor.nonzero)

```python
torch.Tensor.nonzero(*, as_tuple=False)
```

### [paddle.Tensor.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nonzero_cn.html#cn-api-tensor-search-nonzero)

```python
paddle.Tensor.nonzero(as_tuple=False)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| as_tuple      | as_tuple     | bool 类型表示输出数据的格式，默认 False 时，输出一个张量，True 时输出一组一维张量。  |
