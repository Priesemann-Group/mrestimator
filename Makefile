# to compile locally with gitlab-runner:
# gitlab-runner exec docker pages
all: html

html: FORCE
	$(MAKE) -C docs html

pypi:
	python3 ./setup.py sdist
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# pip install -U --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mrestimator

clean: FORCE
	rm -rf __pycache__ */__pycache__
	$(MAKE) -C docs clean

FORCE:
