try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='muvr-ml-sensorcnn',
    version='1.0-SNAPSHOT',
    packages=['train', 'eval'],
    url='https://github.com/muvr/muvr-ml',
    license='',
    author='muvr',
    author_email='muvr@muvr.fitness',
    description='CNN',
    requires=['numpy', 'neon']
)
