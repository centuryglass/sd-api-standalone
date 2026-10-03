#!/bin/bash
# Run pylint to identify code style issues.
pylint --rcfile=.pylintrc intrapaint_api tests examples scripts | less
