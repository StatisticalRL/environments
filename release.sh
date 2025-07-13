
pip freeze > requirements_full.txt
## Remove unused packages from the list to generate requirements.txt

git tag -a v1.2025.07 -m "Initial version"
## Don't forget to push tags (not by default in git push):  git push origin --tags or tick "push tags' in PyCharm