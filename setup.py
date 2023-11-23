from setuptools import setup, find_packages

VERSION = '0.0.5'
DESCRIPTION = 'Python Event Financial Study'
LONG_DESCRIPTION = 'Python package to conduct basic event financial study'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="easy_es", 
        version=VERSION,
        author="Vladislav Pyzhov",
        author_email="vladpyzhov@gmail.com",
        license='MIT',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "pandas",
            "numpy",
            "scipy",
            "plotly",
            "scikit-learn",
            "dataclasses",
            "yfinance",
            "typing-extensions",
            "tqdm",
            "statsmodels"
        ], 
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ],
        project_urls={
            'Source': 'https://github.com/Darenar/easy-event-study',
        }
)