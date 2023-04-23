# 本地开发指南

本文将指导你如何在本地进行代码开发

## 代码要求
- 代码注释请遵守 [Doxygen](http://www.doxygen.nl/) 的样式。
- 所有代码必须具有单元测试。
- 通过所有单元测试。
- 请遵守[提交代码的一些约定](./code_review_cn.html)。

以下教程将指导你提交代码。
## [Fork](https://help.github.com/articles/fork-a-repo/)

跳转到[PaddlePaddle](https://github.com/PaddlePaddle/Paddle) GitHub 首页，然后单击 `Fork` 按钮，生成自己目录下的仓库，比如 <https://github.com/USERNAME/Paddle>。

## 克隆（Clone）

将远程仓库 clone 到本地：

```bash
➜  git clone https://github.com/USERNAME/Paddle
➜  cd Paddle
```


## 创建本地分支

Paddle 目前使用[Git 流分支模型](http://nvie.com/posts/a-successful-git-branching-model/)进行开发，测试，发行和维护，具体请参考 [Paddle 分支规范](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/others/releasing_process.md)。

所有的 feature 和 bug fix 的开发工作都应该在一个新的分支上完成，一般从 `develop` 分支上创建新分支。

使用 `git checkout -b` 创建并切换到新分支。

```bash
➜  git checkout -b my-cool-stuff
```

值得注意的是，在 checkout 之前，需要保持当前分支目录 clean，否则会把 untracked 的文件也带到新分支上，这可以通过 `git status` 查看。

## 开始开发

在本例中，我删除了 README.md 中的一行，并创建了一个新文件。

通过 `git status` 查看当前状态，这会提示当前目录的一些变化，同时也可以通过 `git diff` 查看文件具体被修改的内容。

```bash
➜  git status
On branch test
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)

    test

no changes added to commit (use "git add" and/or "git commit -a")
```

## 编译

关于编译 PaddlePaddle 的源码，请参见[从源码编译](../../../install/compile/fromsource.html) 选择对应的操作系统。

## 单测

`python/paddle/fluid/tests/unittests/` 目录下新增的 `test_*.py` 单元测试会被自动加入工程进行编译。

注意事项：

- **运行单元测试测时需要编译整个工程**，并且编译时需要打开`WITH_TESTING`。

- **执行单测一定要用 ctest 命令**，<font color="#FF0000">不可直接`python test_*.py`</font>。

参考上述[编译](#编译)过程，编译成功后，在`build`目录下执行下面的命令来运行单元测试：

执行:

```bash
ctest -R test_mul_op -V
```

## 提交（commit）

接下来我们取消对 README.md 文件的改变，然后提交新添加的 test 文件。

```bash
➜  git checkout -- README.md
➜  git status
On branch test
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    test

nothing added to commit but untracked files present (use "git add" to track)
➜  git add test
```

Git 每次提交代码，都需要写提交说明，这可以让其他人知道这次提交做了哪些改变，这可以通过`git commit -m "This is description"` 完成。

```bash
➜  git commit -m "This is description"
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
clang-format.......................................(no files to check)Skipped
[my-cool-stuff c703c041] add test file
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 233
```

可以看到，在执行`git commit`后，输出了一些额外的信息。这是使用`pre-commmit`进行代码风格检查的结果，关于代码风格检查的使用问题请参考[代码风格检查指南](./codestyle_check_guide_cn.html)。

## 保持本地仓库最新

在准备发起 Pull Request 之前，需要同步原仓库（<https://github.com/PaddlePaddle/Paddle>）最新的代码。

首先通过 `git remote` 查看当前远程仓库的名字。

```bash
➜  git remote
origin
➜  git remote -v
origin  https://github.com/USERNAME/Paddle (fetch)
origin  https://github.com/USERNAME/Paddle (push)
```

这里 origin 是我们 clone 的远程仓库的名字，也就是自己用户名下的 Paddle，接下来我们创建一个原始 Paddle 仓库的远程主机，命名为 upstream。

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

获取 upstream 的最新代码并更新当前分支。

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

## Push 到远程仓库

将本地的修改推送到 GitHub 上，也就是 https://github.com/USERNAME/Paddle。

```bash
# 推送到远程仓库 origin 的 my-cool-stuff 分支上
➜  git push origin my-cool-stuff
```
