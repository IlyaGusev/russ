from setuptools import find_packages, setup


setup(
    name='russ',
    packages=find_packages(),
    version='0.0.2',
    description='Russian words stress detection',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/russ',
    download_url='https://github.com/IlyaGusev/russ/archive/0.0.2.tar.gz',
    keywords=['nlp', 'russian', 'stress'],
    install_requires=[
        'torch >= 1.10.0',
        'transformers >= 4.17.0',
        'pygtrie >= 2.2'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',

        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: Russian',

        'Programming Language :: Python :: 3.7',
    ],
)
