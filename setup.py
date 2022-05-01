from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read()


setup(
    name="t5_text_to_sql",
    version="1.0",
    install_requires=install_requires,
)

python -m nltk.downloader "punkt"
