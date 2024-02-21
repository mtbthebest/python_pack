#!/bin/bash
chown -R $USER:$USER /env_dl/*
service ssh restart 
/bin/bash