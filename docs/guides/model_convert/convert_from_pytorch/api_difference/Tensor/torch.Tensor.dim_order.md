## [ 组合替代实现 ] torch.Tensor.dim_order

### [torch.Tensor.dim_order](https://pytorch.org/docs/stable/generated/torch.Tensor.dim_order.html?highlight=dim_order#torch.Tensor.dim_order)

```python
torch.Tensor.dim_order()
```

Paddle 无此 API，需要组合实现。获取张量在内存中的物理布局，PaddlePaddle 的 Tensor 默认是 contiguous 的, 因此可直接返回一个从 0 到 Tensor 的维度长度的列表即可。

### 转写示例

```python
# PyTorch 写法
y = x.dim_order()

# Paddle 写法
y = tuple([i for i in range(len(x.shape))])
```
