from setuptools import setup

setup(
    name='scDREAMER',
    version='0.0.0',    
    description='scDREAMER Python package',
    url='https://github.com/Zafar-Lab/scDREAMER',
    author='Hamim Zafar',
    author_email='Hamim@iitk.ac.in',
    license='BSD 2-clause',
    packages=['scDREAMER'],
    install_requires=["numpy==1.21.5",
                        "pandas==1.3.5",
                        "scanpy==1.9.1",
                        "scikit_learn==1.1.1",
                        "scipy==1.7.3",
                        "tables==3.7.0",
                        "tensorflow==1.15.0"                     
                      ],

    classifiers=[
        'Programming Language :: Python :: 3.7'
    ],
)