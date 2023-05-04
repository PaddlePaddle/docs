## [仅 torch 参数更多]torch.logcumsumexp

### [torch.logcumsumexp](https://pytorch.org/docs/1.13/generated/torch.logcumsumexp.html#torch-logcumsumexp)

```python
torch.logcumsumexp(input, dim, *, out=None)
```

### [paddle.logcumsumexp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/logcumsumexp_cn.html#logcumsumexp)

```python
paddle.logcumsumexp(x, axis=None, dtype=None, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                |
| ------- | ------------ | ------------------------------------------------------------------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。                                |
| dim     | axis         | 表示需要计算的维度，仅参数名不一致。                                |
| out     |              | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。                 |
|         | dtype        | 表示输出 Tensor 的数据类型，Pytorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### out：指定输出

```python
# Pytorch 写法
torch.logcumsumexp(x,dim=0, out=output)

# Paddle 写法
paddle.assign(paddle.logcumsumexp(x,axis=0), output)
```
