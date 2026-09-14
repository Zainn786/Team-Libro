import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'erc_tiago_integration'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.json')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
        (os.path.join('share', package_name, 'templates'), glob('templates/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team Libro',
    maintainer_email='team@example.invalid',
    description='Team autonomy adapters and safe integration checks for the ERC TIAGo Pro.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'inspect_books = erc_tiago_integration.inspect_books:main',
            'footprint_guard = erc_tiago_integration.footprint_guard:main',
            'navigation_regression = erc_tiago_integration.navigation_regression:main',
            'solution = erc_tiago_integration.solution:main',
            'motion_smoke_test = erc_tiago_integration.motion_smoke_test:main',
            'navigate_relative = erc_tiago_integration.navigate_relative:main',
            'navigate_named = erc_tiago_integration.navigate_named:main',
        ],
    },
)
