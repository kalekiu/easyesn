from setuptools import setup, find_packages

setup(name='easyesn',
      version='0.1.4.2',
      description='',
      url='https://github.com/kalekiu/easyesn',
      author='Roland Zimmermann, Luca Thiede',
      author_email='support@flashtek.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'progressbar2',
            'dill',
            'multiprocess',
            'sklearn'
      ],
      classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Programming Language :: Python :: 3'
      ],
      zip_safe=False)

# release new package with: python setup.py sdist upload -r pypi