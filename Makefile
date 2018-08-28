all: html

html: FORCE
	$(MAKE) -C docs html

clean: FORCE
	rm -rf __pycache__
	$(MAKE) -C docs clean

FORCE:
