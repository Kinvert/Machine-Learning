from setuptools import setup

package_name = 'mnist_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Claude Apparently',
    maintainer_email='claude@apparently.com',
    description='Simple MNIST classification with ROS2 and webcam inference',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mnist_classifier = mnist_ros2.mnist_classifier_node:main',
            'draw_number = mnist_ros2.draw_number_node:main',
        ],
    },
)
