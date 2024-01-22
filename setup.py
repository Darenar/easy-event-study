from setuptools import setup, find_packages

VERSION = '0.0.9'
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
            "pandas==1.3.5",
            "numpy==1.21.6",
            "scipy==1.7.3",
            "plotly==5.18.0",
            "scikit-learn==1.3.2",
            "yfinance==0.2.33",
            "typing-extensions==4.9.0",
            "tqdm==4.66.1",
            "statsmodels==0.12.2",
            "pytest==7.4.3",
            "python-dateutil==2.8.2"
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