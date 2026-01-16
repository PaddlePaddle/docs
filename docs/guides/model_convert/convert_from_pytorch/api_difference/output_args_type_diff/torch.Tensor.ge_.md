## [ 返回参数类型不一致 ]torch.Tensor.ge_
### [torch.Tensor.ge_](https://pytorch.org/docs/stable/ge_nerated/torch.Tensor.ge_.html)
```python
torch.Tensor.ge_(other)
```

### [paddle.Tensor.greater_equal_]()
```python
paddle.Tensor.greater_equal_(y)
```

返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型。

### 参数映射

| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 输入的 Tensor ，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。 |
|  返回值  |  返回值  | 返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型，需要转写。                                     |
### 转写示例
#### other/返回值
```python
# PyTorch 写法
result = x.ge_(other=2)

# Paddle 写法
dtype = x.dtype
result = x.greater_equal_(y=paddle.to_tensor(2)).cast_(dtype)
```
