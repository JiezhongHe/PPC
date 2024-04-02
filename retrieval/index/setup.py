from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = [Extension("ppc_index",
                 sources=["ppc_index.pyx"],
                 include_dirs=["lib"],
                )]

setup(ext_modules=cythonize(ext, language_level=3))

