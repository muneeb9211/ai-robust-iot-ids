from setuptools import setup, find_packages

setup(
    name="iot-adversarial-ids",
    version="1.0.0",
    description="AI-Powered IoT Intrusion Detection System with Adversarial Robustness Testing",
    author="Muhammad Amar Sohail",
    author_email="amarsohail838@gmail.com",
    url="https://github.com/amarsohail/iot-adversarial-ids",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
