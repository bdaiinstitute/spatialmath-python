.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test - run all unit tests"
	@echo " make coverage - run unit tests and coverage report"
	@echo " make docs - build Sphinx documentation"
	@echo " make dist - build dist files"
	@echo " make upload - upload to PyPI"
	@echo " make clean - remove dist and docs build files"
	@echo " make help - this message$(BLACK)"

test:
	pytest

coverage:
	coverage run --source='spatialmath' -m pytest
	coverage report
	coverage html
	open htmlcov/index.html

docs: .FORCE
	(cd docs; make html)

view:
	open docs/build/html/index.html

dist: .FORCE
	#$(MAKE) test
	python -m build
	ls -lh dist

upload: .FORCE
	twine upload dist/*

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist build

