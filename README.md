# stage_stylo_magique_2024

Internship "Stylo Magique"  (SIMV, IGN).

Segment Anything adaptation to bi-temporal change detection.



[Details](https://ignf.sharepoint.com/sites/SIMV/SitePages/Stage-stylo-magique-2024.aspx?&OR=Teams-HL&CT=1726836826197&clickparams=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIxNDE1LzI0MDgxNzAwNDIxIiwiSGFzRmVkZXJhdGVkVXNlciI6ZmFsc2V9)

Team : 
* Martin Dizier (intern)
* Samy Khelifi (SIMV)
* Nicolas Gonthier (SIMV, LASTIG)



### Installation 
* Create env :
`mamba env create -f environment.yml`
* Activate env `mp`
* Create `.env` at the project root
* Install repo packages : `pip install -e .`


Checkpoints SAM : 
* default or vit_h: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
* vit_b: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).

### Run

```
python src/train.py experiment=<experiment_name> sam_type=small data=levir-cd trainer=gpu compile=False task_name=<name> 
```
* experiment_name : Adaptation method with fusion module
Possible value of experiment_name : see configs/experiment/

* task_name : for directory logs
### Set environment variable
create `.env` file and put project paths :

```
PROJECT_PATH = <..>
DATA_PATH = <..>
SAM_DATA_DEMO_PATH =<..> # useless let it empty
CHECKPOINTS_PATH = <..>
LOGS_PATH = <..>
PYTHONPATH=<..>
SLURM_JOB_ID=0 # get JZ job id for log directory name
HYDRA_FULL_ERROR=1
```

python = 3.10
