from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("process_yolo_detections.pyx"),
)
