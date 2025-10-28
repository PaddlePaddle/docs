## [ 仅 API 调用方式不一致 ]torch.Tensor.argwhere

### [torch.Tensor.argwhere](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.argwhere)

```python
torch.Tensor.argwhere()
```

### [paddle.Tensor.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/nonzero_cn.html#paddle/Tensor/nonzero_cn#cn-api-paddle-Tensor-nonzero)

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
