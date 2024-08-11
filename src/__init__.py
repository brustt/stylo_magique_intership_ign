# load env variables
from datetime import datetime
import os
import re
from dotenv import load_dotenv

 # use https://github.com/ashleve/rootutils instead

local_space = f"/home/MDizier"
pattern = re.compile(local_space)

try:
    match = re.search(pattern, os.getcwd()).group()
    load_dotenv()
    # set local dir logs based on date _${now:%Y-%m-%d}_${now:%H-%M-%S}
    os.environ["SLURM_JOB_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

except:
    print("Welcome in JZAY")
    # need to export env variable in slurm script
    pass
