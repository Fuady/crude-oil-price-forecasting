from setuptools import setup, find_packages

setup(
    name="oil_price_forecast",
    version="1.0.0",
    description="Crude Oil Price Forecasting & Trading Signal Generation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
