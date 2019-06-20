from setuptools import find_packages, setup


setup(
    name='russ',
    packages=find_packages(),
    version='0.0.1',
    description='Russian words stress detection',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/russ',
    download_url='https://github.com/IlyaGusev/russ/archive/0.0.1.tar.gz',
    keywords=['nlp', 'russian', 'recurrent neural networks'],
    package_data={
        'russ': [
            'models/ru-main/config.json',
            'models/ru-main/vocabulary/*',
            'models/ru-main/best.th'
        ]
    },
    install_requires=[
        'torch>=1.0.0',
        'allennlp>=0.8.2',
        'pygtrie>=2.2',
        'pytest-cov>=2.6.1',
        'codecov>=2.0.15'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',

        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: Russian',

        'Programming Language :: Python :: 3.6',
    ],
)
