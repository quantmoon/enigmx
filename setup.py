
from distutils.core import setup
setup(
  name = 'enigmx',         # How you named your package folder (MyLib)
  packages = ['enigmx'],   # Chose the same as name
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here httpshelp.github.comarticleslicensing-a-repository
  description = 'enigmx package',   # Give a short description about your library
  author = 'Quantmoon Technologies',                   # Type in your name
  author_email = 'info@quantmoon.tech',      # Type in your E-Mail
  url = 'quantmoon.tech',   # Provide either the link to your github or to your website
  install_requires=[        # I get to this in a second
          "pandas>=0.22.0",
          "numpy>=1.18.2",
          "zarr",
          "fracdiff"
      ],
  classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
  ],
  python_requires = ">=3.5",
)