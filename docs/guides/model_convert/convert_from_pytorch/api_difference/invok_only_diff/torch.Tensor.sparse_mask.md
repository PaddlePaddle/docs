## [ 仅 API 调用方式不一致 ]torch.Tensor.sparse_mask

### [torch.Tensor.sparse\_mask](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sparse_mask.html#torch.Tensor.sparse_mask)

```python
torch.Tensor.sparse_mask(mask)
```

### [paddle.sparse.mask\_as](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/mask_as_cn.html#paddle.sparse.mask_as)

```python
paddle.sparse.mask_as(x, mask, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
out = x.sparse_mask(coo)

# Paddle 写法
out = paddle.sparse.mask_as(x, coo)
```
