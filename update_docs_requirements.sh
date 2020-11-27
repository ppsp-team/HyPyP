REQUIREMENTS_PATH=docs/requirements.txt

poetry export -f requirements.txt --dev --without-hashes -o $REQUIREMENTS_PATH
dephell deps convert
