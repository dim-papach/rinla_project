from setuptools import setup, find_packages

setup(
    name="fyf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "click",
        "colorama",
        "scikit-image",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "fyf=fyf.cli:main",
        ],
    },
    author="FYF Contributors",
    author_email="dipapachristopoulos@protonmail.com",
    description="Fill Your FITS - Process astronomical FITS images using R-INLA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dim-papach/rinla_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.6",
)
