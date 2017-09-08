import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

__version__ = "unknown"

# "import" __version__
for line in open("micarray/__init__.py"):
    if line.startswith("__version__"):
        exec(line)
        break


# See http://pytest.org/latest/goodpractises.html
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="sfa-numpy",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'numpy!=1.11.0',  # https://github.com/sfstoolbox/sfs-python/issues/11
        'scipy',
    ],

    author="SFA Toolbox Developers",
    author_email="sfstoolbox@gmail.com",
    description="Sound Field Analysis Toolbox",
    long_description=open('README.rst').read(),
    license="MIT",
    keywords="acoustics sound-field-analysis beamforming".split(),
    url="https://github.com/spatialaudio/sfa-numpy",
    platforms='any',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],

    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    zip_safe=True,
)
