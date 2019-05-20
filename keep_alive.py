#! /opt/carnd_p3/behavioral/bin/python3
from workspace_utils import active_session
import os
import subprocess

with active_session():
    # do long-running work here    
    os.system("/opt/carnd_p3/linux_sim/linux_sim.x86_64")
    print("hello")
   