from setuptools import setup, find_packages

setup(
    name='Ssound',
    version='0.2.3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    author='Nick',
    description='A library for audio processing and analysis',
    url='https://github.com/NikolayStrepetov/Ssound.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
