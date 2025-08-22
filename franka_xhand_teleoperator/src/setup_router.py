#!/usr/bin/env python3

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup

def main():
    # Define the extension module
    ext_modules = [
        Pybind11Extension(
            "vr_message_router",
            [
                "vr_message_router.cpp",
            ],
            include_dirs=[
                # Path to pybind11 headers
                pybind11.get_include(),
            ],
            language='c++',
            cxx_std=17,
        ),
    ]

    setup(
        name="vr_message_router",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        python_requires=">=3.8",
    )

if __name__ == "__main__":
    main()