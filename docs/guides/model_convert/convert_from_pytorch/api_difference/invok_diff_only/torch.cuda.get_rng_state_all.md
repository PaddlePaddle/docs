## [ 仅 API 调用方式不一致 ]torch.cuda.get_rng_state_all

### [torch.cuda.get_rng_state_all](https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state_all.html#torch.cuda.get_rng_state_all)

```python
torch.cuda.get_rng_state_all()
```

### [paddle.get_rng_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/get_rng_state_cn.html#paddle/get_rng_state_cn#cn-api-paddle-get_rng_state)

```python
paddle.get_rng_state(device=None)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.cuda.get_rng_state_all()

# Paddle 写法
result = paddle.get_rng_state()
```
