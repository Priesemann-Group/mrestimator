all: html

html: FORCE
	$(MAKE) -C docs html

FORCE:
