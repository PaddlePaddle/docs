## [ torch 参数更多 ]torch.lerp
### [torch.lerp](https://docs.pytorch.org/docs/stable/generated/torch.lerp.html#torch.lerp)
```python
torch.lerp(input,
          end,
          weight,
          *,
          out=None)
```

### [paddle.lerp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/lerp_cn.html#paddle.lerp)
```python
paddle.lerp(x,
            y,
            weight,
            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

|    PyTorch        | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|  input  |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  end  |  y  | 表示输入的 Tensor ，仅参数名不一致。  |
|  weight  |  weight  | 表示输入的 Tensor 。  |
|  out  | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |



### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.lerp(input1,input2,0.5, out=y)

# Paddle 写法
paddle.assign(paddle.lerp(input1,input2, 0.5), y)
```
