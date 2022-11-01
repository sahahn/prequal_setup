#!/bin/sh

# Add to bashrc
echo -e "alias prequal='python3 `pwd`/proc_dataset.py'" >> ~/.bashrc

# Replace shell with new shell, s.t., new command is added this run
exec bash

echo "You may now use command prequal from anywhere."
echo "Warning: if you move this folder, you must re-run this script or the command will break,"
echo "this is because we are justed adding prequal as an alias to the longer command prequal_setup/python3/proc_dataset.py ... "

