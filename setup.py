import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="orthopoly",
    version="0.9.5",
    author="Mark Baum <markmbaum@protonmail.com>",
    author_email="markmbaum@protonmail.com",
    description="Python functions for orthogonal polynomials and (real, 2D, orthonormal) spherical harmonics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markmbaum/orthopoly",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy>=1.5.4'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
