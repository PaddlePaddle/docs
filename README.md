# Fluid Documentation Skeleton

## Build

To build documentation, you need have a linux machine and have python2, virtualenv, gmake installed.

### Preparation

You need to create a `virtualenv` instead of polute the global python library path

```bash
virtualenv .env
```

You can enter virtualenv by

```bash
source .env/bin/activate
```

You can exit virtualenv by

```bash
deactivate
```

### Install dependencies

```bash
# enter virtualenv
source .env/bin/activate
# install dependencies
pip install -r requirements.txt
```

### Make HTML

```bash
# make clean  # make clean to regenerate toctree. Just `make html` may have a cache.
make html
```
and the html files will be generated to `build/html`. You can open `build/html/index.html` with your browser to see the documentation.

## Edit

### Edit documentation

It is suggested to use `reStructuredText` because it is the only official markup language supportted by our documentation generating system, sphinx. `markdown` can also be used. However, since the `markdown` has so many dialects, there is no guarantee that the `markdown` source file can be rendered well.

The `reStructuredText` cheatsheet is [here](http://docutils.sourceforge.net/docs/user/rst/quickref.html).


### Edit structure

The `sphinx` (our documentation generating system) uses `toctree` to organize documentation. `toctree` means `table of content tree`. 

Please see the [sphinx documentation](http://www.sphinx-doc.org/en/master/), especially [`toctree` directives](http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html)
