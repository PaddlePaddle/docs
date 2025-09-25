## [ 组合替代实现 ]torch.device

该 api 有两组参数列表重载，因此有两组差异分析。

-----------------------------------------------

### [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device)

```python
torch.device(type, index)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.device(type='cuda', index=0)

# Paddle 写法
'gpu:0'

# PyTorch 写法
torch.device(type='cpu')

# Paddle 写法
'cpu'
```


-----------------------------------------------

### [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device)

```python
torch.device(device)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# PyTorch 写法
torch.device('cpu')

# Paddle 写法
'cpu'

# PyTorch 写法
torch.device(1)

# Paddle 写法
"gpu:1"

# PyTorch 写法
torch.device("cuda:0")

# Paddle 写法
"gpu:0"
```
