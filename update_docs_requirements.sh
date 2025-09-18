REQUIREMENTS_PATH=docs/requirements.txt

# Check if poetry export plugin is installed
if ! poetry export --help &> /dev/null; then
    echo "poetry-plugin-export not found. Installing..."
    pip install poetry-plugin-export
fi

echo "upgrade pip"
poetry run pip install --upgrade pip
echo "poetry install"
poetry install
poetry export -f requirements.txt --with dev --without-hashes -o $REQUIREMENTS_PATH
echo "hypyp==$(poetry run python -c 'import hypyp; print(hypyp.__version__)')" >> $REQUIREMENTS_PATH
