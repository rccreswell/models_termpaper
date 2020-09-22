# models_termpaper

This repository contains Python scripts for the model misspecification term paper.

Some of the Python functions make use of the DifferentialEquations.jl Julia package (https://diffeq.sciml.ai/stable/). To run these files, use the following steps.

First download and extract Julia, and add the Julia bin to path.

```
$ export PATH=$PATH:.../julia-1.3.1/bin/
```

Then run the python files using the command python-jl rather than python.

```
$ python-jl file.py
```

hERG data was drawn from the following repository: https://github.com/CardiacModelling/hERGRapidCharacterisation
