## [ 返回参数类型不一致 ]torch.Tensor.ne_
### [torch.Tensor.ne\_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.ne_.html#torch.Tensor.ne_)
```python
torch.Tensor.ne_(other)
```

### [paddle.Tensor.not_equal_]()
```python
paddle.Tensor.not_equal_(y)
```

返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型。

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ | ----------------------------------------------- |
| other         | y            | 比较的元素，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。                       |
|  返回值  |  返回值  | 返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型，需要转写。                                     |
### 转写示例
#### other/返回值：比较的元素
```python
# PyTorch 写法
y = x.ne_(other=2)

# Paddle 写法
dtype = x.dtype
y = x.not_equal_(y=paddle.to_tensor(2)).cast_(dtype)
```
