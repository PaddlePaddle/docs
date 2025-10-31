## [ 仅 API 调用方式不一致 ]torch.Tensor.conj_physical

### [torch.Tensor.conj_physical](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.conj_physical)

```python
torch.Tensor.conj_physical()
```

### [paddle.Tensor.conj](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor/conj_cn.html#paddle/Tensor/conj_cn#cn-api-paddle-Tensor-conj)

```python
paddle.Tensor.conj(name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = src.conj_physical()

# Paddle 写法
result = src.conj()
```
