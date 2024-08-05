from setuptools import find_packages
from setuptools import setup

# Read requirements.txt, ignore comments
try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" in line:
            line = line[: line.find("#")].strip()
        if line:
            REQUIRES.append(line)
except FileNotFoundError:
    print("'requirements.txt' not found!")
    REQUIRES = list()

# Read README for the long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = ""


setup(
    name='RL-ADN',
    version='0.1.3',
    packages=find_packages(),
    package_data={
        'rl_adn': [
            'data_sources/network_data/node_123/*.csv',
            'data_sources/network_data/node_25/*.csv',
            'data_sources/network_data/node_34/*.csv',
            'data_sources/network_data/node_69/*.csv',
            'data_sources/time_series_data/*.csv'
        ],
    },
    install_requires=[

    ],
    author='Hou Shengren, Gao Shuyi, Pedro Vargara',
    author_email='houshengren97@gmail.com',
    description='RL-ADN: A Benchmark Framework for DRL-based Battery Energy Arbitrage in Distribution Networks',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ShengrenHou/RL-ADN",
    license='MIT',  # or any license you're using
    keywords='DRL energy arbitrage',
)
