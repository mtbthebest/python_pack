#!/bin/bash
# tmux new-session -d -s rtb123
# tmux send-keys '. /env_nlp/bin/activate && jupyter-lab --port 9000 --allow-root --ip=0.0.0.0 --LabApp.token="" --no-browser' C-m
# tmux detach -s rtb123
service ssh restart 
# tmux new-session -d -s rtb123
# tmux send-keys '. /env_nlp/bin/activate && jupyter-lab --port 9000 --allow-root --ip=0.0.0.0 --LabApp.token="" --no-browser'
# tmux detach -s rtb123
. /env_nlp/bin/activate && jupyter-lab --port 9000 --allow-root --ip=0.0.0.0 --LabApp.token="" --no-browser
# sudo su ubuntu
/bin/bash