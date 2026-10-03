REQUIREMENTS_PATH=docs/requirements.txt

echo "Exporting docs requirements..."
uv export --no-hashes --only-group docs --output-file $REQUIREMENTS_PATH
