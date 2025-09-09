from setuptools import setup, find_packages
import os

setup(
    name="family_linkage_models",
    version="2.0.0",
    packages=["family_linkage_models"],
    include_package_data=True,
    package_data={
        'family_linkage_models.database': ['*.sql'],
    },
    
    install_requires=[
    "pandas>=2.0.0",
    "numpy>=1.24.0", 
    "scikit-learn==1.4.2",
    "psycopg2-binary>=2.9.0",
    "sqlalchemy>=2.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "pytest>=8.0.0",
    "pyarrow>=14.0.0",
    "joblib>=1.2.0",
    "tqdm>=4.60.0",
],
    
    python_requires=">=3.12",
    
    author="Abhinav Pundir",
    description="Family Linkage Models Package",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
