## [ 仅 API 调用方式不一致 ]torch.Tensor.matrix_exp

### [torch.Tensor.matrix_exp](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.matrix_exp)

```python
torch.Tensor.matrix_exp()
```

### [paddle.linalg.matrix_exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/matrix_exp_cn.html#paddle/linalg/matrix_exp_cn#cn-api-paddle-linalg-matrix_exp)

```python
paddle.linalg.matrix_exp(x, name=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = x.matrix_exp()

# Paddle 写法
result = paddle.linalg.matrix_exp(x)
```
