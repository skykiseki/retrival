# -*- coding: utf-8 -*-
from setuptools import setup

LONGDOC = """
我家还蛮大的, 欢迎你们来我家van.

https://github.com/skykiseki/retrival

retrival
====

    关于bm25召回的包都是集成的,而且很慢, 所以自己写了个包用, 
    
    基于Numba加速

完整文档见 ``README.md``

GitHub: https://github.com/skykiseki/retrival
"""

setup(name='retrival',
      version='1.0.0',
      description='Just for BM25',
      long_description=LONGDOC,
      long_description_content_type="text/markdown",
      author='Wei, Zhihui',
      author_email='evelinesdd@qq.com',
      url='https://github.com/skykiseki/retrival',
      license="MIT",
      classifiers=[
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Natural Language :: Chinese (Traditional)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Text Processing',
          'Topic :: Text Processing :: Indexing',
          'Topic :: Text Processing :: Linguistic',
      ],
      python_requires='>=3.6',
      install_requires=[
          'scikit-learn>=1.0',
          'jieba',
          'indxr',
          'numba>=0.54.1',
      ],
      keywords='retrival',
      packages=['retrival'],
      package_dir={'retrival': 'retrival'}
      )
