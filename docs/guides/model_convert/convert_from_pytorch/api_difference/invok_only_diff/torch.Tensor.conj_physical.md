## [ 仅 API 调用方式不一致 ]torch.Tensor.conj_physical

### [torch.Tensor.conj\_physical](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.conj_physical.html#torch.Tensor.conj_physical)

```python
torch.Tensor.conj_physical()
```

### [paddle.Tensor.conj](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#conj-name-none)

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
