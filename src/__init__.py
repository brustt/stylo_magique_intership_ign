# load env variables
import os
import re
from dotenv import load_dotenv

local_space = f"/home/MDizier"
pattern = re.compile(local_space)

try:
    match = re.search(pattern, os.getcwd()).group()
    load_dotenv()
except:
    print("Welcome in JZAY")
    # need to export env variable in slurm script
    pass
