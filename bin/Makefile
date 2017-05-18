# DOE-GPU rights
#
# Makefile to generate and to clean ARCC's Python bytecode files
#

.PHONY: all clean

all:
	python2.7 -mcompileall .

clean:
	rm -rf `find . -name "*.pyc"`

