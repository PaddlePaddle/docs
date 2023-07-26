## [ torch 参数更多 ]torch.fix
### [torch.fix](https://pytorch.org/docs/stable/generated/torch.fix.html?highlight=torch+fix#torch.fix)

```python
torch.fix(input,
          *,
          out=None)
```

### [paddle.trunc](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/trunc_cn.html)

```python
paddle.trunc(input,
             name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|   input       |  input  | 表示输入的 Tensor。  |
|  out  | - |  表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.fix(input, out=y)

# Paddle 写法
paddle.assign(paddle.trunc(input), y)
```
