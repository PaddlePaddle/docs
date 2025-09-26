## [ 无参数 ] torch.Tensor.is_sparse_csr

### [torch.Tensor.is_sparse_csr](https://pytorch.org/docs/stable/generated/torch.Tensor.is_sparse_csr.html)

```python
torch.Tensor.is_sparse_csr
```

### [paddle.Tensor.is_sparse_csr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html)

```python
paddle.Tensor.is_sparse_csr()
```

两者功能一致，但使用方式不一致，前者可以直接访问属性，后者需要调用方法，具体如下：

### 转写示例

```python
# torch 版本可以直接访问属性
x.is_sparse_csr

# Paddle 版本需要调用
x.is_sparse_csr()
```
