
format:
	poetry run black --line-length=100 .

format-check:
	poetry run flake8 --config=.flake8

lint: format format-check

install:
	poetry install

test:
	poetry run coverage run --source=mads_dl_exam2 -m pytest -v -p no:warnings
	poetry run coverage report -m --skip-covered

docker:
	docker build --tag=mads_dl_exam2 --memory=2g .
