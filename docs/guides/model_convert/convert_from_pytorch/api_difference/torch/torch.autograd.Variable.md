## [ 组合替代实现 ]torch.autograd.Variable

### [torch.Tensor.addcmul_](https://pytorch.org/docs/stable/autograd.html#variable-deprecated)
```python
torch.autograd.Variable(data, requires_grad=False)
```

用于自动求导的类，它封装了一个 torch.Tensor 并记录了关于它的操作历史，以便后续进行梯度计算。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例

```python
# PyTorch 写法
torch.autograd.Variable(data, requires_grad=False)

# Paddle 写法
data.stop_gradient = not False
```
