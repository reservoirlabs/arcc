###############################################################################
#
# $Id$
#
###############################################################################

# IRAD rights

# F30602-03-C-0033 rights
PACKAGE_PREFIX=com.reservoir

# These packages will be made into separate jar files
PACKAGE_NAMES=arcc

JAR_FILE_PREFIX=rstream_

DOC_TITLE="Libraries"

JAVAC_OPTS=-source 1.8 -Xlint:unchecked -Wunchecked
JAVAC=javac

.PHONY: arcc_py_compile arcc_py_clean

EXTRA_COMPILE_TARGETS += arcc_py_compile
CLEAN_TARGETS += arcc_py_clean

include ../config/GNUmakefile.rules

# Clean testing results
clean_tests:
