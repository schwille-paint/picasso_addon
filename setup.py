from setuptools import setup

setup(
   name='picasso_addon',
   version='0.1.1',
   description='Some addons for picasso',
   license="Max Planck Institute of Biochemistry",
   author='Stehr Florian',
   author_email='stehr@biochem.mpg.de',
   url="http://www.github.com/schwille-paint/picasso_addon",
   packages=['picasso_addon'],
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=[
		"picasso @ git+https://github.com/jungmannlab/picasso.git#egg=picasso-0.3.0",
   ],
)