"""Setup script for Sonic Stems Splitter."""

from setuptools import setup, find_packages

setup(
    name="sonic-stems-splitter",
    version="0.1.0",
    description="High-performance audio stem splitter",
    author="cryptik093-pixel",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "librosa>=0.11",
        "numpy>=1.26",
        "scipy>=1.13",
        "soundfile>=0.13",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "demucs>=4.0.1",
    ],
    entry_points={
        "console_scripts": [
            "sonic-stems=separation_engine:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
