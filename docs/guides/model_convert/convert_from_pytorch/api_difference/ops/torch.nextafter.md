## [ torch 参数更多 ]torch.nextafter

### [torch.nextafter](https://pytorch.org/docs/2.0/generated/torch.nextafter.html?highlight=nextafter#torch.nextafter)

```python
torch.nextafter(input,
                other,
                *,
                out=None)
```

### [paddle.nextafter](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nextafter_cn.html)

```python
paddle.nextafter(x,
                 y,
                 name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor，仅参数名不一致。  |
| other |  y  | 表示输入的 Tensor，仅参数名不一致。  |
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.nextafter(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]), out=y)

# Paddle 写法
paddle.assign(paddle.nextafter(paddle.to_tensor([1.0, 2.0]),paddle.to_tensor([2.0, 1.0])), y)
```
