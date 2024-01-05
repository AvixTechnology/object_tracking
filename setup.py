from setuptools import setup
import os
package_name = 'object_detection_avix'
# Define the path to your engine file
engine_file = os.path.join(package_name, 'yolov8n.engine')


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), [engine_file]),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='avix',
    maintainer_email='alexmanson_lu@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = object_detection_avix.tracking_node_v2:main',
        ],
    },
)
