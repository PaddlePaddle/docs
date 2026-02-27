## [ 仅 API 调用方式不一致 ]torch.Tensor.is_sparse

### [torch.Tensor.is\_sparse](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_sparse.html#torch.Tensor.is_sparse)

```python
torch.Tensor.is_sparse
```

### [paddle.Tensor.is\_sparse](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#is-sparse)

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
