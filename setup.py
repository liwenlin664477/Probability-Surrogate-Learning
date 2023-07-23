from setuptools import setup, find_packages

setup(
    name="Probability-Surrogate-Learning",
    version="1.1",
    packages=find_packages(),
    author="Wenlin Li",
    author_email="u1327012@utah.edu",
    description="This is a comprehensive machine learning library, "
                "specifically tailored for surrogate learning and active learning.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liwenlin664477/Probability-Surrogate-Learning",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
