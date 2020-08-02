.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test - run all unit tests"
	@echo " make coverage - run unit tests and coverage report"
	@echo " make docs - build Sphinx documentation"
	@echo " make docupdate - upload Sphinx documentation to GitHub pages"
	@echo " make dist - build dist files"
	@echo " make upload - upload to PyPI"
	@echo " make clean - remove dist and docs build files"
	@echo " make help - this message$(BLACK)"

test:
	python -m unittest
ifeq ($(shell which travis),)
	@echo travis not found (gem install travis)
else
	#travis lint .travis.yml
endif

coverage:
	coverage run --omit=\*/test_\* -m unittest
	coverage report

docs: .FORCE
	(cd docs; make html)

docupdate: docs
	cp -r docsrc/build/html/. docs
	git add docs
	git commit -m "rebuilt docs"
	git push origin master

dist: .FORCE
	$(MAKE) test
	python setup.py sdist

upload: .FORCE
	twine upload dist/*

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist

