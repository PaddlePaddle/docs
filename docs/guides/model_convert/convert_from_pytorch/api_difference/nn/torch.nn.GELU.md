## [参数不一致]torch.nn.GELU

### [torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#torch.nn.GELU)

```python
torch.nn.GELU(approximate='none')
```

### [paddle.nn.GELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GELU_cn.html)

```python
paddle.nn.GELU(approximate=False, name=None)
```

其中功能一致, 输入参数用法不一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                                                 |
| ----------- | ------------ | ------------------------------------------------------------------------------------ |
| approximate | approximate  | 是否使用近似计算，PyTorch 取值 none 和 tanh，Paddle 取值为 bool 类型，需要转写。 |

### 转写示例

#### approximate 参数：是否使用近似计算

```python
# 取值为 tanh，PyTorch 写法:
m = torch.nn.GELU(approximate='tanh')
y = m(x)

# Paddle 写法:
m = paddle.nn.GELU(approximate=True)
y = m(x)

# 取值为 none，PyTorch 写法:
m = torch.nn.GELU(approximate='none')
y = m(x)

# Paddle 写法:
m = paddle.nn.GELU(approximate=False)
y = m(x)
```
