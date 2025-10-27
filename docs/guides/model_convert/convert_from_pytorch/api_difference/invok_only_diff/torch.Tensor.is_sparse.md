## [ 仅 API 调用方式不一致 ]torch.Tensor.is_sparse

### [torch.Tensor.is_sparse](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.is_sparse)

```python
torch.Tensor.is_sparse
```

### [paddle.Tensor.is_sparse](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/is_sparse_cn.html#paddle/Tensor/is_sparse_cn#cn-api-paddle-Tensor-is_sparse)

```python
paddle.Tensor.is_sparse()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.is_sparse

# Paddle 写法
result = a.is_sparse()
```
