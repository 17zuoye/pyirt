from setuptools import setup
# from Cython.Build import cythonize

setup(
    name='pyirt',
    version="0.3.2",
    packages=['pyirt',
              'pyirt/solver',
              'pyirt/util', ],
    license='MIT',
    description='A python implementation of EM IRT, specializing in large and sparse data set',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    include_package_data=True,
    url='https://github.com/junchenfeng/pyirt',
    download_url='https://github.com/junchenfeng/pyirt/archive/v0.3.2.tar.gz',
    keywords=['IRT', 'EM', 'big data'],
    zip_safe=False,
    platforms='any',
    install_requires=['numpy',
                      'scipy',
                      'cython',
                      'six',
                      'pymongo',
                      'tqdm'],

    package_data={'pyirt': ["*.pyx"]},

    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
    ],
)
