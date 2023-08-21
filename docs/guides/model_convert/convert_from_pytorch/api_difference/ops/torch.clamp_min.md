## [ torch 参数更多 ]torch.clamp_min
### [torch.clamp_min]()

```python
torch.clamp_min(input,
            min=None,
            *,
            out=None)
```

### [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clip_cn.html#clip)

```python
paddle.clip(x,
            min=None,
            max=None,
            name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input       |  x             | 表示输入的 Tensor ，仅参数名不一致。  |
| min         | min            | 表示裁剪的最小值。                                      |
| -         | max            | 表示裁剪的最大值。Pytorch 无此参数， Paddle 保持默认即可。                                      |
|  out        | -              | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.clamp_min(input, min=-0.5, out=y)

# Paddle 写法
paddle.assign(paddle.clip(input, min=-0.5), y)
```
