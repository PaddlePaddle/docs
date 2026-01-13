## [ torch 参数更多 ]torch.inverse
### [torch.inverse](https://pytorch.org/docs/stable/generated/torch.inverse.html?highlight=inverse#torch.inverse)
```python
torch.inverse(input, *, out=None)
```

### [paddle.inverse](https://github.com/PaddlePaddle/Paddle/blob/5721d267e434c18fa64ff2b99839c7cb6d4cc04d/python/paddle/tensor/math.py#L2900)
```python
paddle.inverse(x, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input          |  x             | 输入的 Tensor ，仅参数名不一致。                                     |
|  out            | -                                       | 表示输出的 Tensor，Paddle 无此参数，需要转写。              |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.inverse(torch.tensor([[2., 0.], [0., 2.]]), out=y)

# Paddle 写法
paddle.assign(paddle.inverse(paddle.to_tensor([[2., 0], [0, 2.]])), y)
```
