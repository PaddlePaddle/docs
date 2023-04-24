## [torch 参数更多]torch.trunc
### [torch.trunc](https://pytorch.org/docs/1.13/generated/torch.trunc.html?highlight=torch+trunc#torch.trunc)

```python
torch.trunc(input,
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
|   input       |  x  | 表示输入的 Tensor ，仅参数名不一致。  |
|  out  | - |  表示输出的 Tensor ， Paddle 无此参数，需要进行转写。    |

### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.trunc([-0.4, -0.2, 0.1, 0.3], out=y)

# Paddle 写法
paddle.assign(paddle.trunc([-0.4, -0.2, 0.1, 0.3]), y)
```
