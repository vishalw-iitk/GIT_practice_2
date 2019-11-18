"""
    Setup for apache beam pipeline.
"""
import setuptools


NAME = 'utils'
VERSION = '1.0'
REQUIRED_PACKAGES = [
    'apache-beam[gcp]',
    'tensorflow==1.14.0',
    'opencv-python',
    'gcsfs',
    'workflow',
    'imutils',
    'mtcnn',
    'dlib'
    ]

setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True
)
