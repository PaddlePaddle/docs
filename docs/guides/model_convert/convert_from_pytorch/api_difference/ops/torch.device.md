## [ 组合替代实现 ]torch.device

### [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device)

```python
torch.device(type, index)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.device('cuda', 0)

# Paddle 写法
str('gpu:0')


# PyTorch 写法
torch.device('cpu')

# Paddle 写法
str('cpu')

```
