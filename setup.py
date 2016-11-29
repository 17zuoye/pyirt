from setuptools import setup
# from Cython.Build import cythonize

setup(
    name='pyirt',
    version="0.2.6",
    packages=['pyirt',
              'pyirt/solver',
              'pyirt/utl', ],
    url='https://github.com/junchenfeng/pyirt',
    license='MIT',
    description='A python implementation of EM IRT, specializing in sparse data',
    author='Junchen Feng',
    author_email='frankfeng.pku@gmail.com',
    include_package_data=True,
    download_url='https://github.com/junchenfeng/pyirt/tarball/0.2.6',
    keywords=['IRT', 'EM', 'big data'],
    zip_safe=False,
    platforms='any',
    install_requires=['numpy',
                      'scipy',
                      'cython'],

    package_data={'pyirt': ["*.pyx"]},
    # ext_modules=cythonize('pyirt/utl/clib.pyx'),

    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.3',
    ],
)
