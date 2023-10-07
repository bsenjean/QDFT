import io
from setuptools import setup, find_packages

setup(
    name='QDFT',
    author='Bruno Senjean',
    author_email='bruno.senjean@umontpellier.fr',
    url='',
    description=('Kohn-Sham Density Functional Theory on Quantum Computers'),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
)
