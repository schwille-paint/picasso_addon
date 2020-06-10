.. _SPT:
	https://github.com/schwille-paint/SPT
.. _lbFCS:
	https://github.com/schwille-paint/lbFCS
   
picasso_addon: Picasso extensions
=================================

This package provides some further functionalities based on `picasso <https://github.com/jungmannlab/picasso>`_ python package including:

- :ref:`Automated minimal net-gradient detection <automng>` for picasso.localize
- :ref:`Automated cluster detection <autopick>` (picks) based on picasso.render
- Easy to use script batch processing

We provide other packages that build up on picasso_addon:

- SPT_  : Complete single particle tracking analysis package (mobile and immobile)
- lbFCS_: For molecular counting and hybridization rate measurements in localization clusters based on DNA-PAINT.

.. image:: files/software-immob.png
    :width: 600px
    :align: center
    :alt: Workflow

picasso_addon was used for data analysis in:

- `Flat-top TIRF illumination boosts DNA-PAINT imaging and quantification <https://www.nature.com/articles/s41467-019-09064-6>`_
- `Toward Absolute Molecular Numbers in DNA-PAINT <https://pubs.acs.org/doi/10.1021/acs.nanolett.9b03546>`_
- `Tracking Single Particles for Hours via Continuous DNA-mediated Fluorophore Exchange <https://www.biorxiv.org/content/10.1101/2020.05.17.100354v1>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   installation
   howto
   modules
   contact

   

