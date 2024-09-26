## [可删除] torch.Tensor.dim_order

### [torch.Tensor.dim_order](https://pytorch.org/docs/stable/generated/torch.Tensor.dim_order.html?highlight=dim_order#torch.Tensor.dim_order)

```python
torch.Tensor.dim_order()
```

获取张量在内存中的物理布局，PaddlePaddle 的 Tensor 默认是 contiguous 的, 因此可直接返回一个从0到 Tensor 的维度长度的列表即可。
