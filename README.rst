picasso_addon
=============

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4792396.svg
   :target: https://doi.org/10.5281/zenodo.4792396
   
This package provides some further functionalities based on `picasso <https://github.com/jungmannlab/picasso>`_ python package including:

- Automated minimal net-gradient detection for picasso.localize
- Automated cluster detection based on picasso.render

Please refer to the `readthedocs <https://picasso-addon.readthedocs.io/en/latest/index.html>`_ for further information.

.. image:: docs/files/software-immob.png
    :width: 300px
    :align: center
    :alt: Workflow

Installation
^^^^^^^^^^^^
To create a `conda <https://www.anaconda.com/>`_ environment with all the necessary dependencies we provide an 
`environment.yml <https://github.com/schwille-paint/picasso_addon/blob/master/environment.yml>`_.
 
To create the environment please open a terminal (Anaconda Prompt on Windows) and type 

.. code-block:: console
    
    (base) conda env create -f=environment.yml
    
This will create a conda environment ``picasso_addon``. To activate the environment type

.. code-block:: console

    (base) activate picasso_addon
    
The picasso_addon package requires the `picasso`_ package. Clone the repository, switch to the downloaded folder and type

.. code-block:: console

    (picasso_addon) python setup.py install
    
This will install the `picasso`_ python package into the `picasso_addon` environment. 

As a last step we will install the `picasso_addon`_ package into the ``picasso_addon`` environment. As before, we clone the repository, switch to the downloaded folder and type

.. code-block:: console

    (picasso_addon) python setup.py install

Done!  


If you don't want to install the picasso_addon package into your environment but you want to be able to import it in any IPython (Spyder, Jupyter) session do the following:
    1. Navigate to ~/.ipython/profile_default
    2. Create a folder called startup if it’s not already there
    3. Add a new Python file called ``start.py`` or modify it and add 
    
.. code-block:: python

    import sys
    sys.path.append('C:/yourpath/picasso_addon')


