from setuptools import setup, find_packages

setup(
    name="tsp-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "tqdm>=4.60.0",
    ],
    author="Jakub Feliński",
    author_email="qbasta@example.com",
    description="Analysis of TSP algorithms",
)