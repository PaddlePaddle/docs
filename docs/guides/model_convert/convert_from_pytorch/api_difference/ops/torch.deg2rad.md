## [ torch 参数更多 ]torch.deg2rad
### [torch.deg2rad](https://pytorch.org/docs/stable/generated/torch.deg2rad.html#torch-deg2rad)

```python
torch.deg2rad(input,
              *,
              out=None)
```

### [paddle.deg2rad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/deg2rad_cn.html#paddle.deg2rad)

```python
paddle.deg2rad(x,
               name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor，仅参数名不一致。  |
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
a = torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
torch.deg2rad(a, out=y)

# Paddle 写法
a = paddle.to_tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]])
paddle.assign(paddle.deg2rad(a), y)
```
