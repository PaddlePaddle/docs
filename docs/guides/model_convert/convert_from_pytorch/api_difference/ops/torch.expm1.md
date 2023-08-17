## [ torch 参数更多 ]torch.expm1
### [torch.expm1](https://pytorch.org/docs/stable/generated/torch.expm1.html?highlight=torch+expm1#torch.expm1)

```python
torch.expm1(input,
            *,
            out=None)
```

### [paddle.expm1](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/expm1_cn.html)

```python
paddle.expm1(x,
             name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.expm1(torch.tensor([-0.4, -0.2, 0.1, 0.3]), out=y)

# Paddle 写法
paddle.assign(paddle.expm1(paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])), y)
```
