## [ 组合替代实现 ]torch.detach

### [torch.detach](https://pytorch.org/docs/stable/autograd.html#variable-deprecated)
```python
torch.detach(input)
```

用于创建一个新的张量，该张量与原始张量共享相同的数据，但不再参与梯度计算。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例

```python
# PyTorch 写法
torch.detach(input=data)

# Paddle 写法
data.detach()
```
