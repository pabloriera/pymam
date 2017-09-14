#!/bin/sh
python setup.py clean
python setup.py install
rm -rf build/

rm -rf dist/doc
sphinx-apidoc . --full -o dist/doc -H 'pymam' -V '1.0'
cd dist/doc

cp -v ../files/index.rst .
cp -v ../files/pymam.rst .
cp -v ../files/conf.py .

make clean
make html

cd ../..
git add dist/\*
git commit -m doc
git subtree push --prefix dist origin gh-pages