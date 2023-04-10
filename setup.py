from setuptools import Extension, setup

ext_modules = [
    Extension(
        "csc.get_path_matrix",
        sources=[
            "src/csc/get_path_matrix.pyx",
        ],
    )
]


if __name__ == "__main__":
    from Cython.Build import cythonize
    import numpy as np

    setup(
        ext_modules=cythonize(ext_modules, language_level="3"),
        include_dirs=[np.get_include()],
    )
