# DOE-GPU rights
#
# Makefile to generate and to clean ARCC's Python bytecode files
#

.PHONY: all clean

all:
	python -mcompileall .

clean:
	rm -rf `find . -name "*.pyc"`

