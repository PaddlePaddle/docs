import sys
import os
import inspect
import ast

# sys.setdefaultencoding('utf-8')

# sys.path.insert(0, os.path.abspath('@PADDLE_BINARY_DIR@/python'))
from recommonmark import parser, transform

import time

try:
    import paddle  # noqa: F401
except:
    print("import paddle error")
breathe_projects = {"PaddlePaddle": "/docs/doxyoutput/xml"}
breathe_default_project = "PaddlePaddle"
MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
AutoStructify.default_config = {
    'enable_auto_doc_ref': False,
    'auto_toc_maxdepth': 1,
    'auto_toc_tree_section': None,
    'enable_auto_toc_tree': True,
    'enable_eval_rst': True,
    'enable_math': True,
    'enable_inline_math': True,
    'commonmark_suffixes': ['.md'],
    'url_resolver': lambda x: x,
    'known_url_schemes': ['http', 'https'],
}

templates_path = ["/templates"]

project = 'PaddlePaddle'
author = '%s developers' % project
copyright = '%d, %s' % (time.localtime(time.time()).tm_year, author)
github_doc_root = 'https://github.com/PaddlePaddle/docs/docs'

# add markdown parser
MarkdownParser.github_doc_root = github_doc_root

os.environ['PADDLE_BUILD_DOC'] = '1'

# Add any Sphinx extension moduleexclude_patterns names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx_sitemap',
    'sphinx.ext.linkcode',
    'recommonmark',
    'sphinx_markdown_tables',
    'breathe',
    'exhale',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'sphinx_design',
]

# exhale
exhale_args = {
    # These arguments are required
    "containmentFolder": "/FluidDoc/docs/inference_api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Inference API",
    "doxygenStripFromPath": "..",
    # "listingExclude": [r"*CMakeLists*", 0],
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT=/FluidDoc/docs/inference_api/paddle_include_file\nMACRO_EXPANSION=NO\nSKIP_FUNCTION_MACROS=YES",
    "verboseBuild": True,
    "generateBreatheFileDirectives": True,
}

cpp_id_attributes = [
    "aligned",
    "prog_filename,",
    "packed",
    "weak",
    "always_inline",
    "noinline",
    "no-unroll-loops",
    "__attribute__((optimize(3)))",
]
cpp_paren_attributes = ["optimize", "__aligned__", "section", "deprecated"]
# primary_domain="cpp"

# nbsphinx
nbsphinx_execute = 'never'

# math
math_number_all = False
mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
    'tex2jax': {
        'inlineMath': [['$', '$'], ["\\(", "\\)"]],
        'processEscapes': 'false',
    },
}
mathjax_path = "https://cdn.bootcss.com/mathjax/2.7.6/MathJax.js"

table_styling_embed_css = True

autodoc_member_order = 'bysource'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.rst': 'restructuredtext',
    #'.txt': 'markdown',
    '.md': 'markdown',
}

# The encoding of source files.
source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index_cn'

html_baseurl = 'https://www.paddlepaddle.org.cn/documentation/docs/'
# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh'
version = ''
# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# html_permalinks_icon='P'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'book/*',
    'design/*',
    '**/*_en.rst',
    '*_en.rst',
    '**/*hidden.*',
    '**/*.en*',
    '*.en*',
    "**/*CMakeLists**",
]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
html_sidebars = {
    '**': ['globaltoc.html'],
}

doc_version = os.environ.get('VERSIONSTR', 'develop')
if len(doc_version) == 0:
    doc_version = 'develop'

GITHUB_REPO_URL = 'https://github.com/PaddlePaddle/Paddle/blob/'
if doc_version != 'develop':
    GITHUB_REPO_URL += 'release/'

# -- Options for HTML output ----------------------------------------------
html_context = {
    'display_github': True,
    'github_user': 'PaddlePaddle',
    'github_repo': 'docs',
    'github_version': 'develop',
    'conf_py_path': '/docs/',
}

if 'VERSIONSTR' in os.environ and os.environ['VERSIONSTR'] != 'develop':
    try:
        float(os.environ['VERSIONSTR'])
        html_context['github_version'] = 'release/' + os.environ['VERSIONSTR']
    except ValueError:
        print(
            "os.environ['VERSIONSTR']={} is not releases's name".format(
                os.environ['VERSIONSTR']
            )
        )
        html_context['github_version'] = os.environ['VERSIONSTR']

# if lang == 'en' and 'pagename' in html_context and html_context['pagename'].startswith('api/'):
#     html_context['display_github'] = False

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # 'canonical_url': '',
    # 'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': 'blob',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 10,
    'includehidden': True,
    'titles_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# -- Options for LaTeX output ---------------------------------------------
latex_engine = 'xelatex'
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    'fncychap': '',
    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r'''\usepackage{ctex}
    ''',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, '%s.tex' % project, project, author, 'manual'),
]

# Use the .. admonition:: directive for Notes sections.
# False to use the .. rubric:: directive instead.
napoleon_use_admonition_for_notes = True
numfig = False
highlight_language = 'python'
html_experimental_html5_writer = False
MARKDOWN_EXTENSIONS = [
    'markdown.extensions.fenced_code',
    'markdown.extensions.tables',
    'pymdownx.superfences',
    'pymdownx.escapeall',
]

math_numfig = False


def change_variable_name(text):
    """
    doc
    :param text:
    :return:
    """
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            lst.append("_")
        lst.append(char)
    return "".join(lst).lower()


def linkcode_resolve(domain, info):
    """
    doc
    :param domain:
    :param info:
    :return:
    """
    if domain != 'py':
        return None
    if not info['fullname']:
        return None
    filename = info['fullname'].replace('.', '/')
    class_names = info['fullname'].split('.')
    api_title = class_names[len(class_names) - 1]
    print(api_title)
    class_name = info['fullname'].replace('.' + api_title, '')
    try:
        current_class = sys.modules[class_name]
        api = getattr(current_class, api_title)
        line_no = None

        if type(api).__name__ == 'module':
            module = os.path.splitext(api.__file__)[0] + '.py'
        else:
            node_definition = (
                ast.ClassDef if inspect.isclass(api) else ast.FunctionDef
            )

            if api.__module__ not in [
                'paddle.fluid.core',
                'paddle.fluid.layers.layer_function_generator',
            ]:
                module = (
                    os.path.splitext(sys.modules[api.__module__].__file__)[0]
                    + '.py'
                )
                with open(module) as module_file:
                    module_ast = ast.parse(module_file.read())

                    for node in module_ast.body:
                        if (
                            isinstance(node, node_definition)
                            and node.name == api_title
                        ):
                            line_no = node.lineno
                            break

                    # If we could not find it, we look at assigned objects.
                    if not line_no:
                        for node in module_ast.body:
                            if isinstance(node, ast.Assign) and api_title in [
                                target.id for target in node.targets
                            ]:
                                line_no = node.lineno
                                break
            else:
                module = os.path.splitext(current_class.__file__)[0] + '.py'
        url = GITHUB_REPO_URL + os.path.join(
            doc_version, 'python', module[module.rfind('paddle') :]
        )
        if line_no:
            return url + '#L' + str(line_no)
        return url
    except Exception as e:
        print("conf.py linkcode_resolve error", e)
        return None


def setup(app):
    """
    doc
    :param app:
    :return:
    """
    # Add hook for building doxygen xml when needed
    # no c++ API for now
    app.add_config_value(
        'recommonmark_config',
        {
            # 'url_resolver': lambda url: github_doc_root + url,
            'enable_math': True,
            'enable_inline_math': True,
            'enable_eval_rst': True,
            'enable_auto_doc_ref': True,
            'auto_toc_tree_section': True,
            'known_url_schemes': ['http', 'https'],
        },
        True,
    )
    app.add_transform(AutoStructify)
