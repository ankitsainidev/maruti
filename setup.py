import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="maruti",
    version="0.0.8",
    author="Ankit Saini",
    author_email="ankitsaini100205@gmail.com",
    description="Maruti Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankitsainidev/maruti",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=['tqdm==4.40.2','opencv-python']
)
