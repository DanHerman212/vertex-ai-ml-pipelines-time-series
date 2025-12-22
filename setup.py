import setuptools

setuptools.setup(
    name='nhits-streaming-pipeline',
    version='0.0.1',
    install_requires=[
        'pandas',
        'numpy',
        'protobuf<5.0.0dev',
        'google-cloud-storage',
        'google-cloud-aiplatform',
        'google-cloud-bigquery',
        'db-dtypes',
        'pyarrow',
        'joblib',
        'scikit-learn',
        'matplotlib',
        'apache-beam[gcp]',
        'requests',
        'google-cloud-pubsub',
        'google-cloud-firestore'
    ],
    packages=setuptools.find_packages(),
)
