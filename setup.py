from setuptools import setup

setup(
    name="Sentiment",
    version="0.1",
    packages=["sentiment"],
    install_requires=[
        'numpy',
        'keras',
        'tensorflow'
    ],
)