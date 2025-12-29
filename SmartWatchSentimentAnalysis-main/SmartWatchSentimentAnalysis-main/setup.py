"""
Setup script for the Smartwatch Sentiment Analyzer project
"""
from setuptools import setup, find_packages

setup(
    name="smartwatch-sentiment-analyzer",
    version="1.0.0",
    description="Sentiment analysis for smartwatch reviews comparing Classical ML vs Transformers",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pandas>=2.1.3",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "transformers>=4.35.2",
        "torch>=2.1.1",
        "datasets>=2.15.0",
        "beautifulsoup4>=4.12.2",
        "requests>=2.31.0",
        "nltk>=3.8.1",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "jinja2>=3.1.2",
        "python-multipart>=0.0.6",
    ],
    python_requires=">=3.8",
)
