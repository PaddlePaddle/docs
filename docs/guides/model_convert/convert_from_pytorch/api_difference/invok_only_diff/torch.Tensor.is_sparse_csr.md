## [ 仅 API 调用方式不一致 ]torch.Tensor.is_sparse_csr

### [torch.Tensor.is\_sparse\_csr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_sparse_csr.html#torch.Tensor.is_sparse_csr)

```python
torch.Tensor.is_sparse_csr
```

### [paddle.Tensor.is\_sparse\_csr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#is-sparse-csr)

```python
paddle.Tensor.is_sparse_csr()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.is_sparse_csr

# Paddle 写法
result = a.is_sparse_csr()
```
