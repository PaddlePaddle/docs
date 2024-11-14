## [ 组合替代实现 ]os.environ.get

### [os.environ.get](https://docs.python.org/zh-cn/3/library/os.html#os.environ)

```python
os.environ.get(key, value)
```

Paddle 无此 API，需要组合实现。

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
