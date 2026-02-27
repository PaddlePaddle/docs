## [ 仅 API 调用方式不一致 ]torch.Tensor.crow_indices

### [torch.Tensor.crow\_indices](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.crow_indices.html#torch.Tensor.crow_indices)

```python
torch.Tensor.crow_indices()
```

### [paddle.Tensor.crows](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#crows)

```python
paddle.Tensor.crows()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = a.crow_indices()

# Paddle 写法
result = a.crows()
```
