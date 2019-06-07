# Initial Makefile
#
#
.PHONY: test0 test clean

test: test0 clean

test0:
	python3 -m unittest tests/test_1d.py
	python3 -m unittest tests/test_3d.py	

clean: 
	@find ./ -maxdepth 2 -name "__pycache__" -type d -exec rm -rf {} \;
