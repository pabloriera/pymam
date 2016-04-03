#!/bin/sh
sudo python setup.py clean
sudo python setup.py install
sudo rm -rf build/

rm -rf dist/doc
sphinx-apidoc . --full -o dist/doc -H 'pymam' -V '1.0'
cd dist/doc

cp -v ../files/index.rst .
cp -v ../files/pymam.rst .
cp -v ../files/conf.py .



make clean
make html

cd ../..
git commit -am doc
git subtree push --prefix dist origin gh-pages
