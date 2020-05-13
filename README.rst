picasso_addon
=============
This package provides some further functionalities based on `picasso <https://github.com/jungmannlab/picasso>`_ python package including:

- Automated net-gradient detection for picasso.localize
- Automated cluster detection (picks) based on picasso.render


Installation
^^^^^^^^^^^^

Prepare conda environment
-------------------------

1. Activate the `conda <https://www.anaconda.com/>`_ environment in which you want picasso_addon to be installed::
	
	conda activate <envname>
	
2. Install all the packages as listed in requirements.txt


Install picasso_addon
---------------------

1. `Clone <https://help.github.com/en/articles/cloning-a-repository>`_ the `picasso_addon <https://github.com/schwille-paint/picasso_addon>`_ repository:: 
	
	git clone  https://github.com/schwille-paint/picasso_addon
	
2. Switch to the cloned folder::
	
	cd picasso_addon
	
3. Install picasso_addon into the environment. This will also install the `picasso <https://github.com/jungmannlab/picasso>`_ python package as dependency:: 
	
	python setup.py install


