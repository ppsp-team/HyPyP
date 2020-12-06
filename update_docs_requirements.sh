REQUIREMENTS_PATH=docs/requirements.txt

echo "upgrade pip"
poetry run pip install --upgrade pip
echo "poetry install"
poetry install
poetry export -f requirements.txt --dev --without-hashes -o $REQUIREMENTS_PATH
echo "hypyp==$(poetry run python -c 'import hypyp; print(hypyp.__version__)')" >> $REQUIREMENTS_PATH
