# Guide of local development

You will learn how to develop programs in local environment under the guidelines of this document.

## Requirements of coding
- Please refer to the coding comment format of [Doxygen](http://www.stack.nl/~dimitri/doxygen/)
- Make sure that option of builder `WITH_STYLE_CHECK` is on and the build could pass through the code style check.
- Unit test is needed for all codes.
- Pass through all unit tests.
- Please follow [regulations of submitting codes](#regulations of submitting codes).

The following guidiance tells you how to submit code.
## [Fork](https://help.github.com/articles/fork-a-repo/)

Transfer to the home page of Github [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) ,and then click button `Fork`  to generate the git under your own file directory,such as <https://github.com/USERNAME/Paddle>。

## Clone

Clone remote git to local:

```bash
➜  git clone https://github.com/USERNAME/Paddle
➜  cd Paddle
```


## Create local branch

At present [Git stream branch model](http://nvie.com/posts/a-successful-git-branching-model/)  is applied to Paddle to undergo task of development,test,release and maintenance.Please refer to [branch regulation of Paddle](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/others/releasing_process.md) about details。

All development tasks of feature and bug fix should be finished in a new branch which is extended from `develop` branch.

Create and switch to a new branch with command `git checkout -b`.


```bash
➜  git checkout -b my-cool-stuff
```

It is worth noting that before the checkout, you need to keep the current branch directory clean, otherwise the untracked file will be brought to the new branch, which can be viewed by  `git status` .


## Use `pre-commit` hook

Paddle developers use the [pre-commit](http://pre-commit.com/) tool to manage Git pre-commit hooks. It helps us format the source code (C++, Python) and automatically check some basic things before committing (such as having only one EOL per file, not adding large files in Git, etc.).

The `pre-commit` test is part of the unit test in Travis-CI. A PR that does not satisfy the hook cannot be submitted to Paddle. Install `pre-commit` first and then run it in current directory：


```bash
➜  pip install pre-commit
➜  pre-commit install
```

Paddle modify the format of C/C++ source code with `clang-format` .Make sure the version of `clang-format` is above 3.8.

Note：There are differences between the installation of `yapf` with `pip install pre-commit` and that with `conda install -c conda-forge pre-commit` . Paddle developers use `pip install pre-commit` 。

## Start development

I delete a line of README.md and create a new file in the case.

View the current state via `git status` , which will prompt some changes to the current directory, and you can also view the file's specific changes via `git diff` .


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

## Build and test

It needs a variety of development tools to build PaddlePaddle source code and generate documentation. For convenience, our standard development procedure is to put these tools together into a Docker image,called  *development mirror* , usually named as `paddle:latest-dev` or `paddle:[version tag]-dev`,such as `paddle:0.11.0-dev` . Then all that need `cmake && make` ,such as IDE configuration,are replaced by `docker run paddle:latest-dev` .

You need to bulid this development mirror under the root directory of source code directory tree

```bash
➜  docker build -t paddle:latest-dev .
```

Then you can start building PaddlePaddle source code with this development mirror.For example,to build a Paddleddle which are not dependent on GPU but in support of AVX commands and including unit test,you can:

```bash
➜  docker run -v $(pwd):/paddle -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TESTING=ON" paddle:latest-dev
```

If you want to build PaddlePaddle based on Python3,you can:

```bash
➜  docker run -v $(pwd):/paddle -e "PY_VERSION=3.5" -e "WITH_FLUID_ONLY=ON" -e "WITH_GPU=OFF" -e "WITH_AVX=ON" -e "WITH_TESTING=ON" paddle:latest-dev
```

Except for the build of PaddlePaddle as `./build/libpaddle.so` and the output of `./build/paddle.deb` file, there is an output of `build/Dockerfile`. What we need to do is to package the PaddlePaddle as a *produce mirror*（ `paddle:prod` ）with following commands.

```bash
➜  docker build -t paddle:prod -f build/Dockerfile .
```

Run all unit tests with following commands:

```bash
➜  docker run -it -v $(pwd):/paddle paddle:latest-dev bash -c "cd /paddle/build && ctest"
```

Please refer to [Installation and run with Docker](../../../beginners_guide/install/install_Docker.html) about more information of construction and test.

## Commit

Next we cancel the modification of README.md,and submit new added test file.

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

It's required that the commit message is also given on every Git commit, through which other developers will be notified of what changes have been made. Type `git commit` to realize it.

```bash
➜  git commit
CRLF end-lines remover...............................(no files to check)Skipped
yapf.................................................(no files to check)Skipped
Check for added large files..............................................Passed
Check for merge conflicts................................................Passed
Check for broken symlinks................................................Passed
Detect Private Key...................................(no files to check)Skipped
Fix End of Files.....................................(no files to check)Skipped
clang-formater.......................................(no files to check)Skipped
[my-cool-stuff c703c041] add test file
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 233
```

<b> <font color="red">Attention needs to be paid：you need to add commit message to trigger CI test.The command is as follows:</font> </b>

```bash
# Touch CI single test of develop branch
➜  git commit -m "test=develop"
# Touch CI single test of release/1.1 branch
➜  git commit -m "test=release/1.1"
```

## Keep the latest local repository

It needs to keep up with the latest code of original repository(<https://github.com/PaddlePaddle/Paddle>）before Pull Request.

Check the name of current remote repository with `git remote`.

```bash
➜  git remote
origin
➜  git remote -v
origin	https://github.com/USERNAME/Paddle (fetch)
origin	https://github.com/USERNAME/Paddle (push)
```

origin is the name of remote repository that we clone,which is also the Paddle under your own account. Next we create a remote host of an original Paddle and name it upstream.

```bash
➜  git remote add upstream https://github.com/PaddlePaddle/Paddle
➜  git remote
origin
upstream
```

Get the latest code of upstream and update current branch.

```bash
➜  git fetch upstream
➜  git pull upstream develop
```

## Push to remote repository

Push local modification to GitHub(https://github.com/USERNAME/Paddle).

```bash
# submit it to remote git the branch my-cool-stuff of origin
➜  git push origin my-cool-stuff
```
