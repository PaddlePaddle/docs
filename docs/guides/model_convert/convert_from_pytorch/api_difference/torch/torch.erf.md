## [ torch 参数更多 ]torch.erf
### [torch.erf](https://pytorch.org/docs/stable/generated/torch.erf.html?highlight=torch+erf#torch.erf)

```python
torch.erf(input,
          *,
          out=None)
```

### [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/erf_cn.html#erf)

```python
paddle.erf(x,
           name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |

### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.erf(torch.tensor([-0.4, -0.2, 0.1, 0.3]), out=y)

# Paddle 写法
paddle.assign(paddle.erf(paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])), y)
```
