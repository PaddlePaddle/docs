## [ 仅 API 调用方式不一致 ]torch.Tensor.col_indices

### [torch.Tensor.col_indices](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.col_indices)

```python
torch.Tensor.col_indices()
```

### [paddle.Tensor.cols](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/cols_cn.html#paddle/Tensor/cols_cn#cn-api-paddle-Tensor-cols)

```python
paddle.Tensor.cols()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.col_indices()

# Paddle 写法
result = a.cols()
```
