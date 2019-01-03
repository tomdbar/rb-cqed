from setuptools import setup

setup(name='rb_cqed',
      version='0.1',
      description='Modelling simple atoms and 87Rb in cavity-QED',
      url='https://github.com/tomdbar/rb_cqed',
      author='Tom Barrett',
      author_email='t.d.barrett91@gmail.com',
      license='MIT',
      packages=['rb_cqed','qutip','dataclasses'],
      python_requires='>=3.6',  # Your supported Python ranges
      include_package_data=True,
      zip_safe=False)