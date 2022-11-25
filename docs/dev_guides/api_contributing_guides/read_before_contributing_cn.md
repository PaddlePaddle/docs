# 贡献前阅读

本章主要介绍开发飞桨原生算子 API 的方法，可先参见通用的 [代码贡献流程](../code_contributing_path_cn.html) 章节，再结合本文介绍的 API 开发要点，即可掌握飞桨原生算子 API 开发方法和流程。

## 一、飞桨原生算子 API 开发解读

飞桨框架 API 前端采用 Python 语言，以便获得更好的编程体验；后端的计算逻辑实现采用 C++ 语言，调用底层算子内核 （kernel）函数实现计算逻辑，以便获得更好的运行性能，如下图所示。

开发一个新的飞桨原生算子，通常需要先开发 C++ 算子，即通过 Yaml 配置定义算子描述、C++ 开发算子 kernel，再封装 Python API；如果要新增的算子可以用其他 Python API 组合得到，则可以只开发 Python API 代码。

- 使用 C++ 定义算子，开发门槛较高，需有一定 C++ 或 CUDA 等软件栈开发基础，但是具有性能优势；
- 使用 Python API 组合方式，只需 Python 编码，代码实现相对简单灵活，但会引入 Python 调度开销，影响性能；如果当前飞桨框架提供的基础算子 API 无法满足需求，仍然需要使用 C++ 实现算子。

<center><img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/api_contributing_guides/images/paddle_api.png?raw=true" width="800" ></center>

> 说明：一般来说 C++ 相比 Python 有着明显的性能优势，所以为了做到模型训练和推理的高效，除了常用的 API 之外，其他包括框架的执行器调度逻辑，算子的 Kernel 实现等都在 C++ 端完成。并且由于目前像 GPU，NPU，XPU 这样的硬件设备有着比 CPU 更强的计算能力，所以深度学习框架也会把算子运算逻辑的 Kernel 在这些硬件设备上进行实现来达到更好的训练和推理性能，但要在这些硬件上开发一些复杂算子的 Kernel 实现成本还是比较高的，因此我们也提供了组合算子的机制（组合式算子）通过复用已有算子 Kernel 来降低新算子 Kernel 的开发成本。


## <span id="apiDesignDoc">二、飞桨 API 设计文档提交说明</span>

设计文档，通常也叫 RFC（Request for Comment）文档，可方便飞桨社区开发者充分交流设计思路，以便进一步完善设计方案，并确保与飞桨设计理念一致。请参考如下步骤完成 API 设计文档的提交：

1. 阅读 [API 设计和命名规范](api_design_guidelines_standard_cn.html)，确保新增 API 符合飞桨相关规范。
2. 根据 [API 设计文档模版](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/api_design_template.md)，填写必要的设计内容。另外可参考 [API 设计文档样例](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20200301_api_design_for_quantile.md)。
3. 将设计文档提交 Pull Request （PR）到 [community/rfcs/APIs/ ](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录下。
4. 等待文档接受评审和讨论，并根据各方意见修改文档。通常飞桨开发者会在三个工作日内回复，如果 API 功能较复杂，还将发起评审会议，并提前在 PR 的评论区公布会议时间、会议地址、参与人、议题等内容，请及时关注 PR 中最新动态。

当设计文档通过评审后，将会合入到  [community/rfcs/APIs/ ](https://github.com/PaddlePaddle/community/tree/master/rfcs/APIs) 目录下。

## 三、飞桨 API 代码实现流程

当 API 设计文档合入后，开发者即可进行代码开发。此过程请参考相应的开发规范，包括如下步骤：

- 如果新增 API 不需要开发新的 C++ 算子，可以用其他 Python API 组合得到新的 API，请参考 [开发 API Python 端](new_python_api_cn.html) 章节完成，包括开发 Python 代码、单元测试代码和 API 文档等步骤。
- 如果新增 API 需要开发新的 C++ 算子，请参考 [开发 C++ 算子](new_cpp_op_cn.html) 章节完成，包括开发算子实现代码、封装 Python API 代码、单元测试代码和 API 文档等步骤。
  - 在 paddle/phi/kernels 目录下存放了飞桨框架已经实现的不同硬件的算子内核，可供开发 C++ 算子 时调用。
  - 有时也需要自己开发新的算子内核，这时可能需要使用硬件支持的软件栈（如 CUDA）来实现，或者使用飞桨框架提供的 Kernel Primitive API 来实现，后者具体介绍请参见 [Kernel Primitive API](../op_optimization/kernel_primitive_api/index_cn.html) 章节。

值得注意的是，代码开发完成后，请确保通过了全部单元测试和 CI 测试，才能合入代码。

<center><img src="https://github.com/PaddlePaddle/docs/blob/develop/docs/dev_guides/api_contributing_guides/images/paddle_api_dev_flow.png?raw=true" width="500" ></center>

## 四、飞桨 API 代码开发规范说明

请遵循如下开发规范和测试要求：

- [代码风格规范](../git_guides/codestyle_check_guide_cn.html)
- [API 设计和命名规范](api_design_guidelines_standard_cn.html)
- [API 单元测试及验收规范](api_accpetance_criteria_cn.html)
- [Paddle CI 测试详解](../git_guides/paddle_ci_manual_cn.html)
