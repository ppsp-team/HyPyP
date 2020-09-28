REQUIREMENTS_PATH=docs/requirements.txt

poetry export -f requirements.txt --dev --without-hashes -o $REQUIREMENTS_PATH
echo "hypyp==$(poetry run python -c 'import hypyp; print(hypyp.__version__)')" >> $REQUIREMENTS_PATH