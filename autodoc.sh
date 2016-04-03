#!/bin/sh
sudo python setup.py clean
sudo python setup.py install

rm -rf dist/doc
sphinx-apidoc . --full -o dist/doc -H 'pymam' -V '1.0'
cd dist/doc

cp -v ../files/pymam.rst .
cp -v ../files/conf.py .


make clean
make html

