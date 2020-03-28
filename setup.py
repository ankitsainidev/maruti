import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="maruti",
    version="1.3.2",
    author="Ankit Saini",
    author_email="ankitsaini100205@gmail.com",
    description="Maruti Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankitsainidev/maruti",
    download_url='https://github.com/ankitsainidev/maruti/archive/v1.3.tar.gz',
    packages=['maruti', 'maruti.vision', 'maruti.deepfake',
              'maruti.torch', 'maruti.imports'],
    package_dir={'maruti': 'maruti'},
    package_data={'maruti': ['deepfake/data/*/*', 'vision/data/*/*']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['tqdm==4.40.2', 'opencv-python', 'facenet_pytorch']
)
