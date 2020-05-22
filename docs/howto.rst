Usage
=====
.. toctree::
   :maxdepth: 2



Localize and undrift
^^^^^^^^^^^^^^^^^^^^

To localize and undrift the raw ome.tif files the following steps are performed:

1. An :ref:`automated minimal net-gradient detection <automng>` is performed to define all valid spots
2. A 2D Gaussian least square fit is performed for every spot using `picasso.localize <https://picassosr.readthedocs.io/en/latest/localize.html>`_
3. The resulting localizations are undrifted by RCC as described in `picasso.render <https://picassosr.readthedocs.io/en/latest/render.html#drift-correction>`_


The `localize_undrift notebook <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/noteboooks/01_localize_undrift.ipynb>`_ guides through the usage of
the picasso_addon.localize.main() function to obtain localized and undrifted localizations lists from raw ome.tif files. If you prefer normal pyhton scripts 
(e.g. for use in spyder) you can find it `here <https://github.com/schwille-paint/picasso_addon/blob/master/scripts/standard/01_localize_undrift.py>`_.

.. _automng:

Automated minimal net-gradient detection
----------------------------------------
Work in progress ...




.. _autopick:

Autopick
^^^^^^^^

To automatically define localization clusters in the rendered localization lists (_render.hdf5) the following steps are performed:

1. The localization list is rendered to a subpixel image given by ``oversampling`` (see `picasso.render <https://picassosr.readthedocs.io/en/latest/render.html>`_).
2. Every subpixel value stands for the number of localizations within each subpixel area.
3. The spot detection function of `picasso.localize <https://picassosr.readthedocs.io/en/latest/localize.html>`_ is employed to define spots (boxes) in the rendered localization image.
4. The number of all localizations within these boxes is calculated and we only consider boxes above a certain threshold ``min_n_locs``.
5. We calculate the center of mass (i.e. localizations) within the remaining boxes. This gives us the pick center coordinates.
6. The pick center coordinates are saved as _autopick.yaml.
7. We employ a KDtree to get all localizations with a distance less than ``pick_diameter/2`` to the pick center coordinates to obtain the picks.
8. Last, we give every localization an ID ``group`` corresponding to its pick identity.