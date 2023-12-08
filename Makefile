lint:
	python3 -m mypy --install-types
	python3 -m mypy src/

fmt:
	python3 -m black src/ scripts/ --exclude src/third_party
	python3 -m isort ./src ./scripts --skip src/third_party
