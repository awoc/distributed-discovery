from setuptools import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="distributed_discovery",
    version="0.0.1",
    description="A process mining library for distributed processes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alexander Collins",
    author_email="alexandercollins00@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="distributed process discovery, pm4py",
    packages=['distributed_discovery',
     'tests',
     'distributed_discovery.objects',
     'distributed_discovery.export',
     'distributed_discovery.discovery',
     'distributed_discovery.visualization',
     'distributed_discovery.visualization.assets',
     'distributed_discovery.util',
     'distributed_discovery.conversion'
    ],
    package_data={'distributed_discovery.visualization.assets': ['envelope.svg', 'envelope_full.svg']},
    include_package_data=True,
    python_requires=">=3.7, <4",
    install_requires=["graphviz", "lxml", "pm4py==2.2.29"],
    extras_require={
        "dev": ["check-manifest", "black"],
        "test": ["coverage", "flake8", "pylint"],
    },
)
