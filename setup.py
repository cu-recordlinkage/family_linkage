from setuptools import setup, find_packages
import os

setup(
    name="family_linkage_models",
    version="2.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'family_linkage_models.database': ['*.sql'],
    },
    
    install_requires=[
        "pandas==2.0.3",
        "numpy==1.25.0", 
        "scikit-learn==1.4.2",
        "psycopg2-binary==2.9.9",
        "sqlalchemy==2.0.17",
        "matplotlib==3.6.0",
        "seaborn==0.12.2",
        "pyyaml==6.0",
        "pytest==8.1.1",
        "pyarrow==14.0.1",
        "joblib==1.2.0",
        "tqdm>=4.60.0", 
    ],
    
    python_requires=">=3.10",
    
    author="Abhinav Pundir",
    description="Family Linkage Models Package",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
