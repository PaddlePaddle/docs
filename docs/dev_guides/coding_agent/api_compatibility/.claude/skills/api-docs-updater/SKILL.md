---
name: api-docs-updater
description: 仅用于《Paddle API 对齐 PyTorch 项目》，负责 Step4，在 API 代码修改完成后，同步更新中文 API 文档，确保文档准确反映 API 的最新行为
allowed-tools: Read Grep Glob‌ Write‌ Edit
context: fork
background: true
verbose: true
disable-model-invocation: false
---

# 一、工作目录说明

涉及以下文档的修改：

| 文档类型 | 文件命名 | 改动点 |
|----------|----------|----------|
| API 概览文档 | `docs/api/paddle/Overview_cn.rst` | API 索引目录，新增 API 时需要更新 |
| API 中文文档 | `{api_name}_cn.rst` | 针对 API 功能改动点，修改文档 |

# 二、工作流程概述

## 基本流程

1. **查找 API 英文文档** - 在两个位置查找：
   - 直接存储在 API 实现代码中的 `__doc__` 文档字符串
   - 集中存储在 `Paddle/python/paddle/_paddle_docs.py` 文件中

2. **对比英文和中文文档** - 识别不一致之处

3. **根据代码修改方案选择对应模式** - 见第三章

4. **按照格式规范更新中文文档** - 见第四章

# 三、常见修改模式

根据代码修改方案的不同，文档需要采用不同的修改模式。本章覆盖所有常见场景，并提供完整的实战示例。

## 模式 1：参数别名（通用装饰器）

**适用场景**：
- 使用 `@param_one_alias` / `@param_two_alias` / `@ParamAliasDecorator` 装饰器
- 仅参数名不同，参数顺序相同
- 常见别名映射：x↔input，y↔other，axis↔dim，keepdim↔keepdims

**修改内容**：
1. 函数签名：无需修改（除非新增 out 参数）
2. 参数部分：每个参数末尾添加别名说明
3. 如果新增 out 参数，添加关键字参数小节

**英文文档要求**（代码 docstring）：
- 格式：`Alias: ` + `` ``别名`` ``
- 多个别名：`Alias: ` `` ``input`` ` or ` `` ``other`` ``
- 位置：参数描述末尾

```python
Args:
    x (Tensor): Input tensor. Alias: ``input``.
    axis (int, optional): Axis. Alias: ``dim``.
```

**中文文档要求**（`.rst` 文件）：
- 格式：`别名 ` + `` ``别名`` ``（与英文格式保持一致）
- 多个别名：`别名 ` `` ``input`` ` ` 或 ` `` ``other`` ``
- 位置：参数描述末尾，句号前

**完整实战示例 - paddle.atan2（参数别名+out 参数）**

```rst
.. py:function:: paddle.atan2(x, y, name=None, *, out=None)

参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor。别名 ``input``。
    - **y** (Tensor) - 输入的 Tensor。别名 ``other``。
    - **name** (str，可选) - 操作的名称。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::

    Tensor：计算结果 Tensor。
```

---

## 模式 2：Overload 重载（专用装饰器）

**适用场景**：
- 使用专用装饰器实现参数顺序转换、参数用法转换、可变参数等
- API 支持两种完全不同的签名（Paddle 风格和 PyTorch 风格）
- 典型场景：参数顺序不同、可变参数、参数用法转换

**修改内容**：
1. 函数签名：保留实现签名（Paddle 风格）
2. 文档正文：明确说明有两种调用方式
3. 参数部分：仍以 Paddle 风格为准
4. 别名说明：对应两种签名的参数

**文档要求**：
- 在正文开头阐述两种签名（Paddle 在前，PyTorch 在后）
- Args/Returns 部分仍以 Paddle 风格签名为准
- 对有别名的参数添加别名说明

**完整实战示例 - paddle.broadcast_tensors**

```rst
.. py:function:: paddle.broadcast_tensors(input, name=None)

此 API 有两种调用方式：

1. ``paddle.broadcast_tensors(input, name=None)`` (Paddle 风格)：
   接收一个 Tensor 序列作为参数

2. ``paddle.broadcast_tensors(*tensors)`` (PyTorch 风格)：
   接收可变个 Tensor 参数

参数
:::::::::
    - **input** (Sequence[Tensor]) - 待广播的 Tensor 序列。别名 ``*tensors``。
    - **name** (str，可选) - 操作名称。

返回
:::::::::
    list[Tensor]：广播后的 Tensor 列表。
```

---

## 模式 3：out 参数支持

**适用场景**：
- 新增 out 参数支持（keyword-only 或位置参数）
- out 为单个 Tensor 或元组

**修改内容**：
1. 函数签名：添加 `, *, out=None`（keyword-only）或 `, out=None`（位置参数）
2. 新增部分：添加关键字参数或参数小节描述 out

**英文文档要求**（代码 docstring）：

keyword-only 形式：
```python
Keyword Args:
    out (Tensor|None, optional): The output tensor. Default: None.
```

位置参数形式：
```python
Args:
    ...
    out (Tensor|None, optional): The output Tensor. Default: None.
```

**中文文档要求**（`.rst` 文件）：

keyword-only 形式：
```rst
.. py:function:: paddle.api(x, name=None, *, out=None)

参数
:::::
    - **x** (Tensor) - 输入 Tensor。
    - **name** (str，可选) - 操作名称。

关键字参数
::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。
```

位置参数形式：
```rst
.. py:function:: paddle.api(x, out=None, name=None)

参数
:::::
    - **x** (Tensor) - 输入 Tensor。
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。
    - **name** (str，可选) - 操作名称。
```

---

## 模式 4：返回值为元组的 out 参数

**适用场景**：
- API 返回多个值（如 `frexp` 返回 mantissa 和 exponent）
- out 参数也是元组类型

**修改内容**：
1. 函数签名：添加 `, *, out=None`
2. 关键字参数：out 类型标注为 `tuple[Tensor, Tensor]`

**完整实战示例 - paddle.frexp**

```rst
.. py:function:: paddle.frexp(x, name=None, *, out=None)

用于把一个浮点数分解为尾数和指数的函数，返回一个尾数 Tensor 和一个指数 Tensor。

参数
:::::::::
    - **x** (Tensor) - 输入是一个多维的 Tensor，数据类型为 float32、float64。别名 ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，默认值为 None。

关键字参数
::::::::::
    - **out** (tuple[Tensor, Tensor]，可选) - 输出 Tensor 元组，若不为 ``None``，计算结果将保存在该 Tensor 元组中，默认值为 ``None``。

返回
:::::::::
mantissa（Tensor）：分解后的尾数，形状和原输入一致。
exponent（Tensor）：分解后的指数，形状和原输入一致。
```

---

## 模式 5：Inplace API 的别名说明

**适用场景**：
- Inplace API（函数名以 `_` 结尾，如 `paddle.abs_`、`paddle.floor_divide_`）
- 通过 Python 装饰器添加参数别名支持
- 仅需补充别名说明，无需修改功能描述

**修改内容**：
1. 函数签名：原样保留，不添加任何修改
2. 新增部分：在 Inplace 说明之后添加 `.. note::` 块描述别名

**中文文档要求**（`.rst` 文件）：

```rst
.. py:function:: paddle.floor_divide_(x, y, name=None)

Inplace 版本的 :ref:`cn_api_paddle_floor_divide` API，对输入 `x` 采用 Inplace 策略。

更多关于 inplace 操作的介绍请参考 `3.1.3 原位（Inplace）操作和非原位操作的区别`_ 了解详情。

.. _3.1.3 原位（Inplace）操作和非原位操作的区别: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id3

.. note::

    别名支持：参数名 ``input`` 可替代 ``x``，参数名 ``other`` 可替代 ``y``，如 ``floor_divide_(input=tensor_x, other=tensor_y)`` 等价于 ``floor_divide_(x=tensor_x, y=tensor_y)``。
```


# 四、格式规范与注意事项

## 格式规范

| 项目 | 规范 | 示例 |
|------|------|------|
| **别名说明位置** | 参数描述末尾，句号前 | `- **x** (Tensor) - 输入的 Tensor。别名 ` ``input``` |
| **别名格式** | 2 个反单引号+别名+2 个反单引号 | `` ``input`` `` 或 `` ``dim`` `` |
| **多个别名** | 用"或"连接 | `别名 ` ``input`` ` 或 ` ``other``` |
| **参数类型** | 可选参数用管道符 | `(float\|None，可选)` 或 `(str\|None，可选)` |
| **关键字参数标题** | "关键字参数"后跟冒号行 | `关键字参数` + 换行 + `:::::::::` |
| **关键字参数缩进** | 4 个空格对齐 | ` ` ` ` `- **out** (Tensor，可选) - ...` |
| **out 参数模板** | 统一说明 | `输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。` |

**rst 格式**：
- 代码块使用 `::`
- 列表项使用 `-` 开头
- 参数类型用 `()` 包裹
- 别名用反引号包裹：`` ``input`` ``

## 注意事项

1. **Tensor 类方法**（如 `paddle.Tensor.abs`）
   - 没有独立文档，无需处理
   - 勿与普通方法（如 `paddle.abs`）混淆

2. **Inplace 方法**（如 `paddle.abs_`）
   - 仅更新代码签名，不需修改文档
   - 参数别名支持与原方法一致

3. **文档内容保持**
   - 保留原有的文档风格和格式
   - 不要大面积删除文档原内容
   - 示例代码采用 COPY-FROM: 格式，不要修改

4. **英文文档与中文文档必须对应**
   - 别名格式完全相同
   - Overload 说明内容对应
   - out 参数描述对齐
