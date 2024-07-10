## [ 仅参数名不一致 ] torch.inference_mode

### [torch.inference_mode](https://pytorch.org/docs/stable/generated/torch.no_grad.html)

```python
torch.inference_mode(mode=True)
```

### [paddle.no_grad](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/no_grad_cn.html)

```python
paddle.no_grad(func=None)
```

inference_mode 会额外禁用视图跟踪和版本计数器，提高推理性能，其他功能一致。此外 mode 参数额外支持 bool 类型，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                      |
| ----------- | ------------ | ----------------------------------------------------------------------------------------- |
| mode   | func      | mode 为函数时，仅参数名不同，作为上下文管理器使用时，mode=True 可忽略该参数，mode=False 时，应移除之(替换为空装饰器) |

### 转写示例
```python
# PyTorch 写法
@torch.inference_mode()
def doubler(x):
    return x * 2

# Paddle 写法
@paddle.no_grad()
def doubler(x):
    return x * 2

# PyTorch 写法
@torch.inference_mode(False)
def doubler(x):
    return x * 2

# Paddle 写法
@paddle_aux.empty_decorator
def doubler(x):
    return x * 2

```
