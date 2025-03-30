from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'trax'
]

setup(
  name='quantum_circuit_optimization',
  version='0.1',
  author='',
  author_email='',
  url='',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  description='Quantum Circuit Optimization Problem',
  requires=[]
)
