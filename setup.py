from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='rb_cqed',
      version='0.1',
      description='Modelling simple atoms and 87Rb in cavity-QED',
      url='https://github.com/tomdbar/rb_cqed',
      author='Tom Barrett',
      author_email='t.d.barrett91@gmail.com',
      license='MIT',
      # packages=['rb_cqed', 'rb_cqed/qutip_patches', 'rb_cqed/tests', 'rb_cqed/atom87rb_params'],
      packages=find_packages(),
      install_requires=[
          'scipy>=1.0',
          'cython>=0.21',
          'qutip>=4.3.1',
          'dataclasses>=0.6',
          'seaborn>=0.8',
          'jupyter'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      python_requires='>=3.6',  # Your supported Python ranges
      include_package_data=True,
      zip_safe=False)