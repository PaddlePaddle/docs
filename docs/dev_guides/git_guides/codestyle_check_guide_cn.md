# 代码风格检查指南

整洁、规范的代码风格，能够保证代码的可读性、易用性和健壮性。Paddle 使用 [pre-commit](http://pre-commit.com/) 工具进行代码风格检查。它可以帮助检查提交代码的不规范问题并格式化（当前会检查C++，Python和CMake语言的代码）；诸如cpplint等工具能提前发现代码的潜在静态逻辑错误，提高开发效率。

在Paddle CI 中，由PR-CI-Codestyle-Check流水线对提交的PR进行代码风格检查，若该流水线执行失败，PR将**无法合入**到Paddle仓库。此时需要根据流水线日志的报错信息，在本地修改代码，再次提交。一般情况下，本地使用`pre-commit`进行代码风格检查的结果和 PR-CI-Codestyle-Check流水线结果是一致的。下面介绍 `pre-commit` 的本地安装与使用方法。

Paddle 目前使用的pre-commit版本是 2.17.0。首先安装并在当前目录运行它：

```bash
➜  pip install pre-commit==2.17.0
➜  pre-commit install
```

>注：通过`pip install pre-commit`和`conda install -c conda-forge pre-commit`安装的`yapf`稍有不同的，Paddle 开发人员使用的是`pip install pre-commit`。

在使用 `git commit` 提交修改时，pre-commit将自动检查修改文件的代码规范，并对不符合规范的文件进行格式化。此时，`git commit` 并未执行成功，需要将pre-commit对文件的修改添加到暂存区，再次commit，直到pre-commit代码检查通过后，本次提交才算完成。
例如，对Paddle/paddle/phi/kernels/abs_kernel.h修改后，提交commit，通过`git diff`查看，会发现clang-format修改了该文件，需要添加修改后，再次`git commit`，完成本次提交。

```bash
➜  git diff
diff --git a/paddle/phi/kernels/abs_kernel.h b/paddle/phi/kernels/abs_kernel.h
index e79216a..7e06204 100644
--- a/paddle/phi/kernels/abs_kernel.h
+++ b/paddle/phi/kernels/abs_kernel.h
@@ -22,4 +22,5 @@ namespace phi {
 template <typename T, typename Context>
 void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

-}  // namespace phi
+void test_func();
+}  // namespace
➜  git add paddle/phi/kernels/abs_kernel.h
➜  git commit -m "test"
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
check for added large files..............................................Passed
check for merge conflicts................................................Passed
check for broken symlinks............................(no files to check)Skipped
detect private key...................................(no files to check)Skipped
fix end of files.........................................................Passed
sort simple yaml files...............................(no files to check)Skipped
clang-format.............................................................Failed
- hook id: clang-format-with-version-check
- files were modified by this hook
cpplint..................................................................Passed
pylint...............................................(no files to check)Skipped
copyright_checker........................................................Passed
cmake-format.........................................(no files to check)Skipped
CMake Lint...........................................(no files to check)Skipped
➜  git diff
diff --git a/paddle/phi/kernels/abs_kernel.h b/paddle/phi/kernels/abs_kernel.h
index 7e06204..c1b803b 100644
--- a/paddle/phi/kernels/abs_kernel.h
+++ b/paddle/phi/kernels/abs_kernel.h
@@ -23,4 +23,4 @@ template <typename T, typename Context>
 void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out);

 void test_func();
-}  // namespace
+}  // namespace phi
➜  git add paddle/phi/kernels/abs_kernel.h
➜  git commit -m "test"
➜  git log
commit xxx
Author: xxx
Date:   xxx

    test

...
```

目前pre-commit主要执行C++, Python, Cmake语言的代码规范和格式化，以及git相关的通用检查和格式化。所有的检查工具信息如下：

|检查工具名称 | 作用 | 当前版本 |
|---|---|---|
|[pre-commit](https://github.com/pre-commit/pre-commit) | hook管理工具 | 2.17.0
|[remove-crlf](https://github.com/Lucas-C/pre-commit-hooks.git) | 将CRLF改为LF | 1.1.14
|[pre-commit-hooks](https://github.com/Lucas-C/pre-commit-hooks.git) | pre-commit自带的hook，执行一些通用检查 | 4.1.0
|[cpplint]((https://github.com/cpplint/cpplint)) |C++代码风格检查 | 1.6.0
|[clang-format]((https://releases.llvm.org/download.html)) | C++代码格式化 | 13.0.0
|[pylint]((https://github.com/PyCQA/pylint/))| python代码风格检查，仅用于检查示例代码 | 2.12.0
|[yapf]((https://github.com/pre-commit/mirrors-yapf))| python代码格式化 | 0.32.0

## FAQ
1. pre-commit==2.17.0要求Python>=3.6.1，建议使用较高版本的Python。
2. 在首次commit时，pre-commit需要初始化环境，执行时间会稍长一些，大概在3min左右。
3. 在首次commit前，请先升级pip，并使用pypi官方镜像源，否则，可能会导致clang-format或者cmake-lint安装失败。命令如下：
```bash
➜  pip install --upgrade pip
➜  pip config set global.index-url https://pypi.python.org/simple

```
