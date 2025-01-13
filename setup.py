from os.path import join, dirname

from setuptools import setup, find_packages


def get_version():
    fname = join(dirname(__file__), "src/eq_detector/__version__.py")
    with open(fname) as f:
        ldict = {}
        code = f.read()
        exec(code, globals(), ldict)  # version defined here
        return ldict['version']


package_name = "eq_detector"

setup(name=package_name,
      version=get_version(),
      description='',
      long_description=open('README.md').read().strip(),
      author='',
      author_email='',
      url='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      py_modules=[package_name],
      install_requires=[
          'numpy',
          'torch',
          'pandas',
      ],
      license='Private',
      zip_safe=False,
      keywords='',
      classifiers=[''],
      package_data={
          package_name: ['py.typed'],
      }
      )
