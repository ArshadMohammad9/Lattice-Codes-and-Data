#!/usr/bin/env bash 
#@ job_name = myjob 
#@ job_type = mpich     
#@ output = output_file  
#@ error  = error_file 
#@ node = 1 
#@ tasks_per_node = 128
#@ class = 128core_new
#@ notification = complete 
#@ notify_user = arshadmd9102@gmail.com 
#@ queue 

python3 Code.py



