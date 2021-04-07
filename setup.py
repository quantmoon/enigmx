import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'enigmx',      
  include_package_data = True,
  packages=setuptools.find_packages(),
  version = '0.0.1',    
  license='MIT',        
  description = 'enigmx package',   
  author = 'Quantmoon Technologies', 
  author_email = 'info@quantmoon.tech',
  url = 'https://www.quantmoon.tech',   
  install_requires=[        
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