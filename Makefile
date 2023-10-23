lint:
	python3 -m mypy --install-types
	python3 -m mypy src/

test:
	python3 -m pytest -vvv -c tests/ -o "testpaths=tests" -W ignore::DeprecationWarning

fmt:
	python3 -m black src/ scripts/ --exclude src/third_party
	python3 -m isort ./src ./scripts --skip src/third_party
