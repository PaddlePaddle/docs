## [ torch 参数更多 ]torch.ceil

### [torch.ceil](https://pytorch.org/docs/stable/generated/torch.ceil.html#torch.ceil)

```python
torch.ceil(input,
           *,
           out=None)
```

### [paddle.ceil](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ceil_cn.html)

```python
paddle.ceil(x,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input  |   x   | 表示输入的 Tensor ，仅参数名不一致。   |
| out | - | 表示输出的结果 Tensor，Paddle 无此参数，需要进行转写。|


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.ceil(torch.tensor([-0.6341, -1.4208]), out=y)

# Paddle 写法
paddle.assign(paddle.ceil(paddle.to_tensor([-0.6341, -1.4208])), y)
```
