from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fp:
  long_description = fp.read()

setuptools.setup(
  name="tfdone",
  version="0.0.1",
  author="Kazufumi Hosoda",
  author_email="hosodakazufumi@gmail.com",
  description="TensorFlow implementation of DONE",
  long_description="TensorFlow implementation of DONE (Direct ONE-shot learning with Hebbian weight imprinting)",
  long_description_content_type="text/markdown",
  url="https://github.com/hosodakazufumi/tfdone",
  packages=find_packages(),
  license="Apache License 2.0",
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.7',
)
