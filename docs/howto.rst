.. _picasso.localize:
	https://picassosr.readthedocs.io/en/latest/localize.html
.. _picasso.render:
	https://picassosr.readthedocs.io/en/latest/render.html#drift-correction>


Usage
=====
.. toctree::
   :maxdepth: 2


Localize and undrift
^^^^^^^^^^^^^^^^^^^^
General description
-------------------
To localize and undrift the raw ome.tif files the following steps are performed:

1. An :ref:`automated minimal net-gradient detection <automng>` is performed to define all valid spots
2. A 2D Gaussian least square fit is performed for every spot using `picasso.localize`_
3. The resulting localizations are undrifted in ``segments`` by RCC as described in `picasso.render`_ 

How to use
----------
The `localize_undrift notebook <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/noteboooks/01_localize_undrift.ipynb>`_ guides through the usage of
the picasso_addon.localize.main() function to obtain localized and undrifted localizations lists from raw ome.tif files. If you prefer normal pyhton scripts 
(e.g. for use in spyder) you can find it `here <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/standard/01_localize_undrift.py>`_.


.. _autopick:

Autopick
^^^^^^^^
General description
-------------------
To automatically define localization clusters in the rendered localizations (_render.hdf5) we go from a pointillistic to a pixel based presentation.
We can then use a modified version of the spot finding algorithm of `picasso.localize`_.
The following steps are performed:

1. The localization list is rendered to a subpixel image given by ``oversampling`` (see `picasso.render`_).
2. Every subpixel value now stands for the number of localizations within each subpixel area.
3. The spot detection function of `picasso.localize`_ is employed to define spots (boxes) in the rendered localization image.
4. The number of all localizations within these boxes is calculated and we only consider boxes above a certain threshold ``min_n_locs``.
5. We calculate the center of mass (i.e. localizations) within the remaining boxes. This gives us the pick center coordinates.
6. The pick center coordinates are saved as _autopick.yaml.
7. We employ a KDtree to get all localizations with a distance less than ``pick_diameter/2`` to the pick center coordinates to obtain the picks.
8. Last, we give every localization an ID ``group`` corresponding to its pick identity. Localizations not corresponding to a pick are disregarded.
9. The result is saved as _picked.hdf5.

How to use
----------
The `autopick notebook <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/noteboooks/02_autopick.ipynb>`_ guides through the usage of
the picasso_addon.autopick.main() function to obtain localization clusters above a certain localization threshold from *_render.hdf5 localization lists. 
If you prefer normal python scripts (e.g. for use in spyder) you can find it `here <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/standard/02_autopick.py>`_.


Detailed info
^^^^^^^^^^^^^

Data structure of localization lists
------------------------------------
Please have a look at this `notebook <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/examples/load_datastructure.ipynb>`_ to learn how to load 
picassos' localization lists and which observables it contains.

.. _automng:

Automated minimal net-gradient detection
----------------------------------------
We will insert a notebook here to explain what is happening in detail, please be patient.