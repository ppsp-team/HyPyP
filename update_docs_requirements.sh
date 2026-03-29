REQUIREMENTS_PATH=docs/requirements.txt

echo "Exporting docs requirements..."
uv export --no-hashes --group dev --output-file $REQUIREMENTS_PATH
echo "hypyp==$(uv run python -c 'import hypyp; print(hypyp.__version__)')" >> $REQUIREMENTS_PATH
