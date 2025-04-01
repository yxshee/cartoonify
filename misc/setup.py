from setuptools import setup, find_packages

setup(
    name="cartoonify",
    version="0.1.0",
    author="Yash Dogra",
    author_email="your.email@example.com",
    description="A tool to transform photos into cartoon-style images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yxshee/Cartoonify",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "tqdm",
        "wandb",
    ],
)
