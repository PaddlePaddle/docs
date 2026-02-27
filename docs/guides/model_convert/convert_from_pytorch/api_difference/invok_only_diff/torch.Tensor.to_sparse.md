## [ 仅 API 调用方式不一致 ]torch.Tensor.to_sparse

### [torch.Tensor.to\_sparse](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to_sparse.html#torch.Tensor.to_sparse)

```python
torch.Tensor.to_sparse(sparse_dim)
```

### [paddle.Tensor.to_sparse_coo](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/to_sparse_coo_cn.html#paddle/Tensor/to_sparse_coo_cn#cn-api-paddle-Tensor-to_sparse_coo)

```python
paddle.Tensor.to_sparse_coo(sparse_dim)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
b = a.to_sparse(1)

# Paddle 写法
b = a.to_sparse_coo(1)
```
