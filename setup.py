from setuptools import setup

setup(name='rb_cqed',
      version='0.1',
      description='Modelling simple atoms and 87Rb in cavity-QED',
      url='https://github.com/tomdbar/rb_cqed',
      author='Tom Barrett',
      author_email='t.d.barrett91@gmail.com',
      license='MIT',
      packages=['rb_cqed'],
      setup_requires=[
          'qutip==4.3.1',
          'dataclasses>=0.6',
          'seaborn>=0.8'
      ],
      install_requires=[
          'qutip==4.3.1',
          'dataclasses>=0.6',
          'seaborn>=0.8'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      python_requires='>=3.6',  # Your supported Python ranges
      include_package_data=True,
      zip_safe=False)