Python package boilerplate (IDU) 
================================

.. logo-start

.. figure:: https://sun9-46.userapi.com/impf/aUFBStH0x_6jN9UhgwrKN1WN4hZ9Y2HMMrXT2w/NuzVobaGlZ0.jpg?size=1590x400&quality=95&crop=0,0,1878,472&sign=9d33baa41a86de35d951d4bbd8011994&type=cover_group
   :alt: The Institute of Design and Urban Studies

.. logo-end

|Documentation Status| |PythonVersion| |Black|

.. readme-start 

Installation
------------

The library can be installed with ``pip``:

::

   pip install git+https://github.com/vasilstar97/python-package-boilerplate@main


How to use
----------

Use the library by importing classes from ``my_package``:

::

   from my_package.my_module_1 import MyClass

For more detailed use case see our `examples <#examples>`__.

Data
----

All the required data is stored in `examples
data folder <./examples/data>`__.

Not published data
------------------

Some heavy and intermediate files stay **local only** and are not pushed
to GitHub (they are listed in ``.gitignore``):

- ``examples/data/vyborg/osm_*.geojson`` — raw OpenStreetMap extracts;
- ``examples/data/vyborg/walk_graph.pkl`` — walkability graph built from
  the OSM data;
- ``examples/cache/`` — calculation cache;
- ``.vscode/``, ``examples/vyborg_pipeline_backup.ipynb`` — editor
  settings and a notebook backup.

Published results in ``examples/data/vyborg/``: ``blocks.geojson``,
``boundaries.geojson``, ``buildings_blocks.geojson``,
``school_blocks.geojson``, ``kindergarten_blocks.geojson``,
``polyclinic_blocks.geojson``, ``vyborg_metrics.geojson`` and all CSV
files (accessibility matrix, provision, density, morphotypes).

To regenerate the missing files, open
``examples/vyborg_pipeline.ipynb``, set ``USE_SAVED_DATA = False`` and
run the cells from top to bottom. Internet access is required to
download the OSM data. Local copies of the files are kept on disk.

Examples
--------

Next examples will help one to get used to the library:

1. Main `example <./examples/my_example.ipynb>`__ of the library.
2. ...

Documentation
-------------

Detailed information and description of the library is available in
`documentation <https://vasilstar97.github.io/python-package-boilerplate/>`__.

Project Structure
-----------------

The latest version of the library is available in the ``main`` branch.

The repository includes the following directories and modules:

-  `my_package <./my_package>`__
   directory with the library code:

   -  ...
   -  ...

-  `tests <./tests>`__
   ``pytest`` testing
-  `examples <./examples>`__
   examples of how the library works
-  `docs <./docs>`__
   documentation sources

Developing
----------

...

License
-------

The project has `BSD-3-Clause license <./LICENSE>`__

Acknowledgments
---------------

...

Contacts
--------

You can contact me:

-  `Vasilii Starikov <https://t.me/vasilstar>`__ - assistant

Also, you are welcomed to the `issues <./issues>`__ section!

Publications
------------

-  `My publication name from scholar or something <https://scholar.google.com/>`__
-  ...

.. |Documentation Status| image:: https://github.com/vasilstar97/python-package-boilerplate/actions/workflows/documentation.yml/badge.svg?branch=main
   :target: https://vasilstar97.github.io/python-package-boilerplate/
.. |PythonVersion| image:: https://img.shields.io/badge/python-3.10-blue
   :target: https://pypi.org/project/geopandas/
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. readme-end
