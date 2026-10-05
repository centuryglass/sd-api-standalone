#!/bin/bash
# Run pylint to identify code style issues.
pylint --rcfile=.pylintrc sd_backend_client tests examples scripts | less
