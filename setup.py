from setuptools import setup, find_packages

setup(name='artitect',
      version='0.0.1',
      include=["artitect*"],
      packages=find_packages(include=['data', 'models', 'artitect']),
      include_package_data=True
      )