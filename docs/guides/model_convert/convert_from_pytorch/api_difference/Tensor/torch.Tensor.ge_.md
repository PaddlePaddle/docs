## [ 输入参数类型不一致 ]torch.Tensor.ge_

### [torch.Tensor.ge_](https://pytorch.org/docs/stable/ge_nerated/torch.Tensor.ge_.html)

```python
torch.Tensor.ge_(other)
```

### [paddle.Tensor.greater_equal_]()

```python
paddle.Tensor.greater_equal_(y)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 输入的 Tensor ，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.ge_(other=2)

# Paddle 写法
result = x.greater_equal_(y=paddle.to_tensor(2))
```
