# to compile locally with gitlab-runner:
# gitlab-runner exec docker pages
all: html

html: FORCE
	$(MAKE) -C docs html

clean: FORCE
	rm -rf __pycache__ */__pycache__
	$(MAKE) -C docs clean

FORCE:
