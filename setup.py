import setuptools

with open("requirements.txt") as f:
    requirements = f.read().split('\n')

setuptools.setup(
    name="WikipediaQA",
    version="0.1.0",
    author="Savelov Dmitry",
    author_email="savasavelck@mail.ru",
    # description="A AsyncIO-based module for interacting with the Mediawiki API using aiohttp",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url=r"https://github.com/JosephThePatrician/WikipediaQA",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=requirements,
    # classifiers=(
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # )
)
