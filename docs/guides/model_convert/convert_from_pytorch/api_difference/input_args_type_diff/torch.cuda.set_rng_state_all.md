## [ 输入参数类型不一致 ]torch.cuda.set_rng_state_all
### [torch.cuda.set\_rng\_state\_all](https://docs.pytorch.org/docs/stable/generated/torch.cuda.set_rng_state_all.html#torch.cuda.set_rng_state_all)
```python
torch.cuda.set_rng_state_all(new_states)
```

### [paddle.set\_rng\_state](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/set_rng_state_cn.html#paddle.set_rng_state)
```python
paddle.set_rng_state(state_list, device='gpu')
```

其中 PyTorch 与 Paddle 的参数类型不一致，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                                                            |
| ---------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| new_states | state_list   | 表示每个设备需要的状态，PyTorch 类型为 torch.ByteTensor 列表，Paddle 类型为 GeneratorState 列表，需要转写。 |
| -          | device       | 返回随机数生成器状态的设备，Paddle 取值 gpu。                                                           |

### 转写示例
#### new_states 参数
```python
# PyTorch 写法
x = torch.cuda.get_rng_state_all()
torch.cuda.set_rng_state_all(x)

# Paddle 写法
x = paddle.get_rng_state()
paddle.set_rng_state(x, device='gpu')
```
