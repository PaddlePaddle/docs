## [ 仅 API 调用方式不一致 ]torch.cuda.manual_seed

### [torch.cuda.manual_seed](https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html#torch.cuda.manual_seed)

```python
torch.cuda.manual_seed(seed)
```

### [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/seed_cn.html#paddle/seed_cn#cn-api-paddle-seed)

```python
paddle.seed(seed)
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
torch.cuda.manual_seed(123)

# Paddle 写法
paddle.seed(123)
```
