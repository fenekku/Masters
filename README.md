#README

This repository holds the essential files of the thesis.

Each directory has a README to explain its content/purpose and to make sure it is tracked (even if empty of results) by the revision control system.

This is a cleaned copy of the working original code repository. Code may not work out of the box and the repository provides only limited documentation. However the code can be perused and reused.

##Installation

Target: Ubuntu 12.04+

There are two possible setups: system wide and virtual environment. If you are not familiat with virtualenv then go for the system wide implementation.

###System wide


Run

    sudo apt-get install python-<requirement>

for each *requirement* in the requirements.txt file


###Virtual Environment

Run

    sudo apt-get build-dep python-<requirement>

for each *requirement* found in the requirements.txt file

Then install each in your virtual environment:

    pip install -r requirements.txt


##License

Copyright 2014 Guillaume Viger

MIT License