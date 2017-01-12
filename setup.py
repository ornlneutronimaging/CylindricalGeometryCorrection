#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name = "neutronbraggedge",
    version = "1.0.1",
    author = "Jean Bilheux",
    author_email = "bilheuxjm@ornl.gov", 
    packages = find_packages("python", exclude=['tests', 'notebooks']),
    package_dir = {'': "python"},
    #package_data = { 'python.neutronbraggedge' : ['material_list.dat']},
    include_package_data = True,
    data_files = [('neutronbraggedge', ['python/neutronbraggedge/config.cfg']),
                  ('neutronbraggedge/data', ['python/neutronbraggedge/data/material_list.dat'])
                  ],
    test_suite = 'tests',
    install_requires = [
        'numpy',
        'configparser',
#        'pandas',
        'lxml',
        'html5lib',
        'beautifulsoup4',
    ],
    dependency_links = [
    ],
    description = "Bragg Edge work at the SNS",
    license = 'BSD',
    keywords = "neutron imaging bragg edge",
    url = "https://github.com/ornlneutronimaging/BraggEdge",
    classifiers = ['Development Status :: 3 - Alpha',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research'
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5'],
    # download_url = '',
)


# End of file
