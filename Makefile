# to compile locally with gitlab-runner:
# gitlab-runner exec docker pages
all: html

html: FORCE
	$(MAKE) -C docs html

pypi:
	python3 ./setup.py sdist
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

clean: FORCE
	rm -rf __pycache__ */__pycache__
	$(MAKE) -C docs clean

FORCE:
