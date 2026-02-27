## [ 仅 API 调用方式不一致 ]torch.Tensor.argwhere

### [torch.Tensor.argwhere](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.argwhere.html#torch.Tensor.argwhere)

```python
torch.Tensor.argwhere()
```

### [paddle.Tensor.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#nonzero-as-tuple-false)

```python
paddle.Tensor.nonzero(as_tuple=False)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = input.argwhere()

# Paddle 写法
result = input.nonzero()
```
