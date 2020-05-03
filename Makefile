.FORCE:

test:
	python -m unittest

coverage:
	coverage run --omit=\*/test_\* -m unittest
	coverage report

docs: .FORCE
	(cd docs; make html)
