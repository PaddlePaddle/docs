## [ 仅 API 调用方式不一致 ]torch.Tensor.col_indices

### [torch.Tensor.col\_indices](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.col_indices.html#torch.Tensor.col_indices)

```python
torch.Tensor.col_indices()
```

### [paddle.Tensor.cols](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#cols)

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
