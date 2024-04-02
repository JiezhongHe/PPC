from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = [Extension("generator",
                 sources=["generator.pyx"],
                 include_dirs=["./lib/"],
                )]

setup(ext_modules=cythonize(ext, language_level=3))