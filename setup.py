import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qiskit-vqe-wrapper",
    version="0.0.1",
    author="Mårten Skogh",
    author_email="marten.skogh@gmail.com",
    description="Small wrapper for Qiskit VQE execution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MartenSkogh/QiskitVQEWrapper"
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
