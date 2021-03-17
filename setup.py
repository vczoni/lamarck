from setuptools import setup, find_packages
from lamarck.version import __version__

VERSION = __version__

requirements = open('requirements.txt', 'r').read().splitlines()

setup(
    name='lamarck',
    packages=find_packages(),
    version=VERSION,
    license='MIT',
    description='Genetic Algorithm Prototype.',
    author='Victor Zoni',
    author_email='vczoni@gmail.com',
    url='https://github.com/vczoni/lamarck',
    download_url=VERSION.join(
        ['https://github.com/vczoni/lamarck/archive/v', '.tar.gz']),
    keywords=['GENETIC', 'ALGORITHM', 'GA', 'BASIC', 'GENERAL',
              'OPTIMIZATION', 'SIMULATION'],
    install_requires=requirements,
    python_requires='>=3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
