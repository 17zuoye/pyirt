from setuptools import setup
# from Cython.Build import cythonize

setup(
    name='pyirt',
    version="0.3",
    packages=['pyirt',
              'pyirt/solver',
              'pyirt/util', ],
    url='https://github.com/junchenfeng/pyirt',
    license='MIT',
    description='A python implementation of EM IRT, specializing in large and sparse data set',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    include_package_data=True,
    download_url='https://github.com/junchenfeng/pyirt/tarball/0.3',
    keywords=['IRT', 'EM', 'big data'],
    zip_safe=False,
    platforms='any',
    install_requires=['numpy',
                      'scipy',
                      'cython',
                      'six',
                      'pymongo'],

    package_data={'pyirt': ["*.pyx"]},

    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
    ],
)
