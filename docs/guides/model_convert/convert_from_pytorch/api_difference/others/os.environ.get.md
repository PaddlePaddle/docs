## [ 组合替代实现 ]os.environ.get

### [os.environ.get](https://docs.python.org/zh-cn/3/library/os.html#os.environ)

```python
os.environ.get(key, value)
```

Paddle 无此 API，需要组合实现。该 API 一般情况下与 Paddle 无关，仅在 torch 分布式相关的深度学习用法里才需转写。可以用来获取参与分布式训练的进程总数，以及获取当前进程的 rank。

### 转写示例

```python
# PyTorch 写法
os.environ.get(key)

# Paddle 写法
if key =="WORLD_SIZE"
    paddle.distributed.get_world_size()
else if key =="LOCAL_RANK"
    padlde.distributed.get_rank()
```
