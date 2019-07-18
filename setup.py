from setuptools import setup

setup(
    name='pyirt',
    version="0.3.4",
    packages=['pyirt',
              'pyirt/solver',
              'pyirt/util', ],
    license='MIT',
    description='A python implementation of Item Response Theory(IRT), specializing in big dataset',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    include_package_data=True,
    url='https://github.com/junchenfeng/pyirt',
    download_url='https://github.com/17zuoye/pyirt/archive/v0.3.4.tar.gz',
    keywords=['IRT', 'EM', 'big data'],
    zip_safe=False,
    platforms='any',
    install_requires=['numpy',
                      'scipy',
                      'cython',
                      'pymongo',
                      'tqdm',
                      'python-decouple'],

    package_data={'pyirt': ["*.pyx"]},

    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
    ],
)
