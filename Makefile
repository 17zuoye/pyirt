init:
	pip install pipenv
	pipenv install --dev
	cp settings.ini.example settings.ini

test:
	nosetests --with-coverage --cover-package=pyirt --cover-html tests

linttest: ## PEP8 compliance
	flake8 pyirt tests *.py
