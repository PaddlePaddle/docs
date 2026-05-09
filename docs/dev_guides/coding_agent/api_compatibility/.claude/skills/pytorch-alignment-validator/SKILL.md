---
name: pytorch-alignment-validator
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step3：Pytorch 对齐验证，基于 PaConvert 工具验证 Paddle API 与 PyTorch API 是否用法完全对齐一致
disable-model-invocation: false
---

# 一、标准工作流程

请严格按以下 Step 依次执行，不要自行修改或跳过 Step：

## Step 1: 标记已对齐的 API
1. 定位文件：`${ROOT_DIR}/PaConvert/paconvert/api_mapping.json`
2. 将已对齐的 PyTorch API 的 Matcher 设置为`ChangePrefixMatcher`，其他字段全部删除掉

> 注意 torch.abs、torch.abs_、torch.Tensor.abs、torch.Tensor.abs_是四个不同的 API

## Step 2: 增加测试用例
**目的：** 判断是否满足如下测试规范，如不满足，则需增加测试用例使之符合规范

**修改位置：**
- Pytorch 单测路径： `${ROOT_DIR}/PaConvert/tests/`
- Pytorch 单测文件名： `test_<API 名称>.py`
- **测试文件命名规范**：将 API 路径中的 `.` 替换为 `_`，Tensor 类保留大写 `T`，最终加上 `test_` 前缀和 `.py` 后缀
  - `torch.argmax` → `test_argmax.py`（顶层函数）
  - `torch.Tensor.argmax` → `test_Tensor_argmax.py`（类方法）
  - `torch.linalg.inv` → `test_linalg_inv.py`（子模块）
- **新增测试 case 策略**：若现有测试不满足规范，应**新增测试 case**（追加到后面），而非修改或删除已有测试

**测试规范：**
1. **参数覆盖要全面**
   测试所有可能的 Pytorch 参数用法：
   - 全部位置参数：`func(a, b, c)`
   - 全部关键字参数：`func(x=a, y=b, z=c)`
   - 混合参数：`func(a, y=b, z=c)`
   - 乱序关键字：`func(z=c, x=a, y=b)`
   - 指定所有默认参数：`func(a, dim=0，offset=1)`
   - 缺省所有默认参数：`func(a)`
   - 变量传参：`args=(a,b); func(*args)`
   - 参数别名：`func(input=a, other=b)`

2. **输入数据要有效**
   - 不能使用全零张量等无效输入
   - 需要包含不同维度的输入（1D、2D、3D 等）
   - 覆盖不同数据类型（float32、float64、int 等）

3. **测试数量要充分**
   - 涉及 N 个新参数时，包含各种排列组合的用法
   - 测试用例越多越好，充分验证功能正确性

**测试用例模板：**
```python
import textwrap
import pytest
from apibase import APIBase

obj = APIBase("torch.target_api")

def test_case_1():
    """Basic usage"""
    pytorch_code = textwrap.dedent("""
        import torch
        result = torch.target_api(torch.tensor([1, 2, 3]))
    """)
    obj.run(pytorch_code, ["result"])

def test_case_2():
    """Positional arguments test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1.0, 2.0])
        result = torch.target_api(x, 2, 1)
    """)
    obj.run(pytorch_code, ["result"])

def test_case_3():
    """Keyword arguments test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1.0, 2.0])
        result = torch.target_api(input=x, dim=1, dtype=torch.float32)
    """)
    obj.run(pytorch_code, ["result"])

def test_case_4():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1.0, 2.0])
        result = torch.target_api(dtype=torch.float32, dim=1, input=x)
    """)
    obj.run(pytorch_code, ["result"])

def test_case_5():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1., 2.], requires_grad=True)
        y = torch.target_api(x)
        y.sum().backward()
        x_grad = x.grad
    """)
    # Skip gradient attribute check because Paddle's stop_gradient mechanism differs from PyTorch's requires_grad mechanism at the framework level.
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)

def test_case_6():
    """Edge case test"""
    pytorch_code = textwrap.dedent("""
        import torch
        result = torch.target_api(torch.empty(0))
    """)
    obj.run(pytorch_code, ["result"])

def test_case_7():
    """Expression argument test"""
    pytorch_code = textwrap.dedent("""
        import torch
        result = torch.target_api(torch.tensor([1, 2, 3]), 1 + 1)
    """)
    obj.run(pytorch_code, ["result"])
```

**特殊场景处理：**

1. **GPU 环境测试**：
```python
import paddle
import pytest

should_skip = not paddle.device.is_compiled_with_cuda()
skip_reason = "Test requires CUDA"

@pytest.mark.skipif(condition=should_skip, reason=skip_reason)
def test_case_with_cuda():
    ...
```

2. **不支持功能标记**：
```python
@pytest.mark.skip(reason="Not supported")
def test_unsupported_case():
    ...
```

3. **自定义比较逻辑**（通常不需要）：
```python
class CustomAPIBase(APIBase):
    def compare(self, name, pytorch_result, paddle_result, ...):
        # 自定义比较逻辑
        pytorch_str = str(pytorch_result).removeprefix("torch.")
        assert pytorch_str == paddle_result
```

## Step 3: 运行单元测试

在补充完测试用例后，确保能通过所有用例，请按以下命令验证执行（不可修改）：

1. 本地执行以下命令：
   ```bash
   cd ${ROOT_DIR}/PaConvert/
   python -m pytest tests/test_<API 名称>.py
   ```

2. 根据报错信息，修改代码或测试用例（禁止通过修改 api_mapping.json 来使单测通过），确保所有测试用例通过。注意每次修改 Paddle 源码后，必须重新编译方可生效：
```bash
cd /workspace/Paddle/build
cmake ..
make -j$(nproc)
```

编译注意事项：
- 无需重装，直接生效（勿执行 setup/install 等安装操作）
- 勿删除 build 目录（否则增量编译失效，编译时间极长）



# 二、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 所有路径使用 `${ROOT_DIR}` 变量表示根目录，需自行替换为实际路径
3. 确保测试用例覆盖所有参数组合和边界情况
4. 验证通过是对齐的金标准，不可跳过
5. 复盘记忆中的历史易错点，避免重复犯错
6. **PaConvert 说明**：Pytorch->Paddle 代码转换工具，可以搭建起 Pytorch-Paddle API 之间的桥梁。

# 三、常见问题处理

## Q1：PyTorch 代码执行失败

**错误标识**：Failed to execute pytorch code

**根本原因**：PyTorch 单元测试代码存在问题，无法正常执行

**处理策略**：修改 PyTorch 单元测试代码，确保能正确执行 Pytorch 代码

**验证标准**：测试代码应该能够在标准的 PyTorch 环境中正常运行

---

## Q2：Paddle 代码执行失败

**错误标识**：Failed to execute paddle code

**根本原因**：Pytorch 单测正确，但修改后的 Paddle API 实现存在问题，导致代码无法执行

**处理策略**：需要返回到前序 Step 修改，因此结束本 Step，将报错信息返回给主控智能体分析

**关联任务**：Paddle API 需要进一步修改以兼容 PyTorch 接口

---

## Q3：计算结果不一致

**错误标识**：Unable to align results

**根本原因**：Pytorch 单测正确，但 Paddle API 与 PyTorch API 计算结果存在差异

**处理策略**：需要返回到前序 Step 修改，因此结束本 Step，将报错信息返回给主控智能体分析

**验证要求**：Paddle API 需要进一步修改以兼容 PyTorch 接口，确保数值精度、数据类型、形状等完全一致

**禁止通过配置 api_mapping.json 为非 ChangePrefixMatcher 来使单测通过**，本步骤的通过标准为：Matcher 配置为 ChangePrefixMatcher + 单测运行通过。

---

## Q4：如何禁用不支持的测试用例

**错误现象**：
Paddle 因为暂不支持某些功能（如类型提升、标量输入等）而报错

**解决方法**：
```python
# 将 def test_case_x(): 改为 def _test_case_x():
# 并添加注释说明原因
def _test_case_2():  # Paddle does not support scalar input
    ...
```

**注意**：仅允许禁用因已知 Paddle 限制导致的失败用例，其他情况不允许禁用

---
