from setuptools import setup

setup(
   name='picasso_addon',
   version='1.0.2',
   description='Some addons for picasso',
   license="MIT License",
   author='Stehr Florian',
   author_email='florian.stehr@gmail.com',
   url="http://www.github.com/schwille-paint/picasso_addon",
   packages=['picasso_addon'],
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=[
		"picasso @ git+https://github.com/jungmannlab/picasso.git",
   ],
)