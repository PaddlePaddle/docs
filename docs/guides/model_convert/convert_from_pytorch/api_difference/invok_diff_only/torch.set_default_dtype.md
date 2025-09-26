## [ 参数完全一致 ] torch.set_default_dtype

### [torch.set_default_dtype](https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html)

```python
torch.set_default_dtype(d)
```

### [paddle.set_default_dtype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/set_default_dtype_cn.html)

```python
paddle.set_default_dtype(d)
```

两者功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | ----------------------------------------------------------------------------------------- |
| d           | d            | 全局默认数据类型，torch 支持 float32 和 float64，paddle 支持 bfloat16、float16、float32 和 float64。  |
