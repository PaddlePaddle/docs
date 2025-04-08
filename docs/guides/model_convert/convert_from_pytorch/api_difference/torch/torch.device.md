## [ 组合替代实现 ]torch.device

### [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device)

```python
torch.device(type, index, device)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.device('cuda', 0)

# Paddle 写法
'gpu:0'


# PyTorch 写法
torch.device('cpu')

# Paddle 写法
'cpu'

# PyTorch 写法
type = 'cuda'
index = 0
torch.device(index=index, type=type)

# Paddle 写法
type = type.replace('cuda', 'gpu')
f'{type}:{index}'

# PyTorch 写法
type = 'cuda'
index = 0
torch.device(index=index, type=type)

# Paddle 写法
type = type.replace('cuda', 'gpu')
f'{type}:{index}'

# PyTorch 写法
a = 1
torch.device(a)

# Paddle 写法
type = f'gpu:{type}'
```
