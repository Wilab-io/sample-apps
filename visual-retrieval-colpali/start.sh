#!/bin/bash
cd src
shad4fast build
if [ ! -f output.css ]; then
    echo "Error: output.css was not generated!"
    exit 1
fi
echo "output.css was generated successfully"
python main.py
