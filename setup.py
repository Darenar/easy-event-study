from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Python Event Financial Study'
LONG_DESCRIPTION = 'Python package to conduct basic event financial study'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="easy_es", 
        version=VERSION,
        author="Vladislav Pyzhov",
        author_email="vladpyzhov@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], 
        keywords=['python'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)