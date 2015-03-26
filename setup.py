from setuptools import setup

setup(
  name = 'pyirt',
  version = '0.2.5',
  packages = ['pyirt',
		'pyirt/solver',
		'pyirt/utl',
		'pyirt/test',],
  url = 'https://github.com/junchenfeng/pyirt',
  license='MIT',
  description = 'A python implementation of EM IRT, specializing in sparse massive data',
  author = 'Junchen Feng',
  author_email = 'frankfeng.pku@gmail.com',
  include_package_data=True,
  download_url = 'https://github.com/junchenfeng/pyirt/tarball/0.2.5',
  keywords = ['IRT', 'EM', 'big data'],
  zip_safe=False,
  platforms='any',
  install_requires=[
   'numpy',
   'scipy',
   'cython',],
  package_data={'pyirt': ["*.pyx"]},
  classifiers = [
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
  ],
)
