## [ 组合替代实现 ]torch.Tensor.is_pinned

### [torch.Tensor.is_pinned](https://pytorch.org/docs/stable/generated/torch.Tensor.is_pinned.html?highlight=is_pinned#torch.Tensor.is_pinned)

```python
torch.Tensor.is_pinned()
```

返回张量是否在固定内存上; Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
y = a.is_pinned(b)

# Paddle 写法
y = 'pinned' in str(a.place)
```
