## [ 仅 API 调用方式不一致 ]torch.numel

### [torch.numel](https://pytorch.org/docs/stable/generated/torch.numel.html)

```python
torch.numel(input)
```

### [paddle.Tensor.size](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/size_cn.html#paddle/Tensor/size_cn#cn-api-paddle-Tensor-size)

```python
paddle.Tensor.size
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.numel(a)

# Paddle 写法
result = a.size
```
