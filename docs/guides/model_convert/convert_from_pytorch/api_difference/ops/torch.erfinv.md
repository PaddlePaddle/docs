## [ torch 参数更多 ]torch.erfinv
### [torch.erfinv](https://pytorch.org/docs/stable/generated/torch.erfinv.html?highlight=torch+erfinv#torch.erfinv)

```python
torch.erfinv(input,
             *,
             out=None)
```

### [paddle.erfinv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erfinv_cn.html)

```python
paddle.erfinv(x,
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
torch.erfinv(torch.tensor([0, 0.5, -1.]), out=y)

# Paddle 写法
paddle.assign(paddle.erfinv(paddle.to_tensor([0, 0.5, -1.], dtype="float32")), y)
```
