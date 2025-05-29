## [ 组合替代实现 ]setuptools.setup

### [setuptools.setup](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)

```python
setuptools.setup(*args, **attrs)
```

Paddle 无此 API，需要组合实现。该 API 一般情况下与 Paddle 无关，仅在与 torch 相关的深度学习用法里才需要转写，用来构建一个包含自定义扩展（如 C++ ）的 PyTorch 包。

### 转写示例

```python
# PyTorch 写法
setuptools.setup(*args, **attrs)

# Paddle 写法
if "cmdclass" in attrs:
    if "paddle.utils.cpp_extension.BuildExtension" in attrs["cmdclass"]:
        paddle.utils.cpp_extension.setup(attrs)
```
