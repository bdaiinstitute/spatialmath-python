from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='spatialmath-python', 

    version='0.1', #TODO 

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    description='Provides spatial maths capability for Python.', #TODO
    
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/petercorke/spatialmath-python',

    author='Peter Corke',

    author_email='rvc@petercorke.com', #TODO

    keywords='python SO2 SE2 SO3 SE3 rotation euler roll-pitch-yaw quaternion transforms robotics vision pose',

    license='', #TODO

    python_requires='>=3.2',

    packages=find_packages(),

    install_requires=['numpy', 'scipy', 'matplotlib']
    
)
