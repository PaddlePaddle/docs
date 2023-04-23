.. _cn_api_distributed_fleet_utils_fs_LocalFS:

LocalFS
-------------------------------

.. py:class:: paddle.distributed.fleet.utils.LocalFS
一个本地文件系统工具类。

代码示例
::::::::::::

COPY-FROM: paddle.distributed.fleet.utils.LocalFS

方法
::::::::::::
ls_dir(fs_path)
'''''''''
列出 `fs_path` 路径下所有的文件和子目录。

**参数**

    - **fs_path** (str)：本地文件路径。

**返回**

    - Tuple，一个包含所有子目录和文件名的 2-Tuple，格式形如：([subdirname1, subdirname1, ...], [filename1, filename2, ...])。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.ls_dir

mkdirs(fs_path)
'''''''''
创建一个本地目录。

**参数**

    - **fs_path** (str)：本地文件路径。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.mkdirs

rename(fs_src_path, fs_dst_path)
'''''''''
重命名本地文件名。

**参数**

    - **fs_src_path** (str)：重命名前原始文件名。
    - **fs_dst_path** (str)：新文件名。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.rename

delete(fs_path)
'''''''''
删除本地文件（或目录）。

**参数**

    - **fs_path** (str)：本地文件路径。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.delete

is_file(fs_path)
'''''''''
判断当前路径是否是一个文件。

**参数**

    - **fs_path** (str)：本地文件路径。

**返回**

    - Bool：若当前路径存在且是一个文件，返回 `True`，反之则返回 `False` 。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.is_file

is_dir(fs_path)
'''''''''
判断当前路径是否是一个目录。

**参数**

    - **fs_path** (str)：本地文件路径。

**返回**

    - Bool：若当前路径存在且是一个目录，返回 `True`，反之则返回 `False` 。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.is_dir

is_exist(fs_path)
'''''''''
判断当前路径是否存在。

**参数**

    - **fs_path** (str)：本地文件路径。

**返回**

    - Bool：若当前路径存在返回 `True`，反之则返回 `False` 。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.is_exist

touch(fs_path, exist_ok=True)
'''''''''
创建一个本地文件。

**参数**

    - **fs_path** (str)：本地文件路径。
    - **exist_ok** (bool)：文件路径已存在时程序是否报错。若 `exist_ok = True`，则直接返回，反之则抛出文件存在的异常，默认不抛出异常。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.touch

mv(src_path, dst_path, overwrite=False)
'''''''''
本地文件移动。

**参数**

    - **src_path** (str)：移动前源文件路径名。
    - **dst_path** (str)：移动后目标文件路径名。
    - **overwrite** (bool)：若目标文件已存在，是否删除进行重写，默认不重写并抛出异常。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.mv

list_dirs(fs_path)
'''''''''
列出本地路径下所有的子目录。

**参数**

    - **fs_path** (str)：本地文件路径。

**返回**

    - List：该路径下所有的子目录名。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.LocalFS.list_dirs
