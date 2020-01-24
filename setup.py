import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="maruti",
    version="0.1.6",
    author="Ankit Saini",
    author_email="ankitsaini100205@gmail.com",
    description="Maruti Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankitsainidev/maruti",
    packages=['maruti'],
    package_dir={'maruti': 'maruti'},
    package_data={'maruti': ['data/*/*','vision/data/*/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['tqdm==4.40.2','opencv-python']
)
