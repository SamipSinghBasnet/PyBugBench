#!/bin/bash

mkdir -p repos
cd repos

# repo-name + GitHub URL
repos=(
  "airflow https://github.com/apache/airflow.git"
  "django https://github.com/django/django.git"
  "fastapi https://github.com/tiangolo/fastapi.git"
  "flask https://github.com/pallets/flask.git"
  "matplotlib https://github.com/matplotlib/matplotlib.git"
  "numpy https://github.com/numpy/numpy.git"
  "pandas https://github.com/pandas-dev/pandas.git"
  "pytest https://github.com/pytest-dev/pytest.git"
  "requests https://github.com/psf/requests.git"
  "scikit-learn https://github.com/scikit-learn/scikit-learn.git"
)

for entry in "${repos[@]}"; do
  name=$(echo $entry | cut -d' ' -f1)
  url=$(echo $entry | cut -d' ' -f2)

  if [ ! -d "$name" ]; then
    echo "Cloning $name..."
    git clone "$url" "$name"
  else
    echo "Repo $name already exists."
  fi
done
