from setuptools import setup
from pathlib import Path
long_description = Path('README.md').read_text()
print(long_description)

setup(
    name='SwarmML',
    version='1.0',
    description='Feature Selection with PSO',
    long_description=long_description,
    author='Saleh Ahmad',
    author_email='salehahmad2106@gmail.com',
    url='https://github.com/SalehAhmad1/SwarmML',
    packages=['SwarmML'],
    install_requires=[
        'sklearn',
        'pandas',
        'numpy',
        'readme_md',
    ],
    long_description_content_type='text/markdown',
    keywords=['feature selection', 'particle swarm optimization', 'PSO','machine learning','data science'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Programming Language :: Python'
    ],
)