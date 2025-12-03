## [ 仅 API 调用方式不一致 ]torch.Tensor.det

### [torch.Tensor.det](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.det)

```python
torch.Tensor.det()
```

### [paddle.linalg.det](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/det_cn.html#paddle/linalg/det_cn#cn-api-paddle-linalg-det)

```python
paddle.linalg.det(x, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.det()

# Paddle 写法
result = paddle.linalg.det(x)
```
