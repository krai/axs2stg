# MLPerf Inference - Text-to-Image (SDXL) task - [TheStageAI](https://app.thestage.ai) implementation

## Prerequisites

### Install system dependencies for building Python v3.10
```
sudo apt-get install \
libffi-dev libbz2-dev liblzma-dev libssl-dev libreadline-dev libsqlite3-dev zlib1g-dev
```

### Setup [KRAI](https://krai.ai) [AXS](https://github.com/krai/axs) environment

#### Clone

Clone the AXS repository:
```
git clone --branch mlperf_5.1 https://github.com/krai/axs ~/axs
```

#### Init

Define environment variables in your `~/.bashrc`:
```
echo "export PATH='$PATH:$HOME/axs'" >> ~/.bashrc
```

#### Test
```
source ~/.bashrc
axs version
```

#### Import public AXS repositories

Import the required public repos ([axs2mlperf](https://github.com/krai/axs2mlperf), [axs2stg](https://github.com/krai/axs2stg)) into your local work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2stg
```

#### Build Python v3.10
```
axs byquery installed,python3,minor_version=10,patch_version=18
```


### Setup [TheStageAI](https://app.thestage.ai) environment

#### Install the Python package

##### Under standard Python environment
```
pip install thestage
```

##### Under externally managed Python environment
```
pipx install thestage
```
<details><summary>installed package thestage 0.6.3, installed using Python 3.12.3</summary><pre>
anton@h100:~$ pipx install thestage
  installed package thestage 0.6.3, installed using Python 3.12.3
  These apps are now globally available
    - thestage
⚠️  Note: '/home/anton/.local/bin' is not on your PATH environment variable. These apps will not be globally accessible until your PATH is updated. Run `pipx ensurepath` to automatically add it, or manually
    modify your PATH in your shell's config file (i.e. ~/.bashrc).
done! ✨ 🌟 ✨
anton@h100:~$ pipx ensurepath
Success! Added /home/anton/.local/bin to the PATH environment variable.
_
Consider adding shell completions for pipx. Run 'pipx completions' for instructions.
_
You will need to open a new terminal or re-login for the PATH changes to take effect.
_
Otherwise pipx is ready to go! ✨ 🌟 ✨
</pre></details>

#### Configure the [API token](https://app.thestage.ai/profile/tokens/active)

```
thestage config set --api-token eyJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJkZWNiZjQyZi1kZTAyLTRjZWEtYmU1MC1lMDU0NWY0OWJlNjgiLCJzdWIiOiJhcGlVc2VyIiwidXNlcklkIjoxNzgsImNsaWVudElkIjoxODksInRva2VuVHlwZSI6ImFjY2VzcyIsIm9hdXRoQ2xpZW50SWQiOiJUSEVTVEFHRV9DTEkiLCJleHAiOjE3ODUyNDQyNzB9.4zFJW_GVZBiJCUkbCngVAOyUsJU5WSEuISZHhFByiZo
```


### Benchmark

The benchmark uses TheStageAI backend with the following key parameters:
* `backend=stageai` - TheStageAI backend for text-to-image inference
* `dtype=fp16` - Data type precision
* `device=cuda` - CUDA device support
* `num_gpus=8` - Number of GPUs to use
* `axs_device_id=0+1+2+3+4+5+6+7` - GPU device IDs

See below commands for the Server/Offline scenarios and under the Accuracy/Performance/Compliance modes (adapted from [actual commands](https://github.com/mlcommons/inference_results_v5.1/tree/main/closed/The_Stage/measurements/h100_x8/stable-diffusion-xl) used for the v5.1 submissions).

#### Offline

##### Accuracy
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18 \
, get accuracy_report
```

##### Performance
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=24072 \
, get performance
```

##### Compliance

###### TEST01
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=24072
```

###### TEST04
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=24072
```

#### Server

##### Accuracy
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18 \
, get accuracy_report
```

##### Performance
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=12036 \
, get performance
```

##### Compliance

###### TEST01
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=12036
```

###### TEST04
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=12036
```

### Generate submission tree

#### Closed Division
```
axs byname submitter , full_run \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=closed
```

#### Open Division

##### Offline only
```
axs byname submitter , full_run \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=open --scenarios,=Offline
```

<details><summary></summary><pre>
[2025-09-10 10:14:28,531 submission_checker.py:3396 INFO] ---
[2025-09-10 10:14:28,531 submission_checker.py:3400 INFO] Results open/The_Stage/results/h100_x8/stable-diffusion-xl/offline 18.125
[2025-09-10 10:14:28,531 submission_checker.py:3402 INFO] ---
[2025-09-10 10:14:28,531 submission_checker.py:3491 INFO] ---
[2025-09-10 10:14:28,531 submission_checker.py:3492 INFO] Results=1, NoResults=0, Power Results=0
[2025-09-10 10:14:28,531 submission_checker.py:3499 INFO] ---
[2025-09-10 10:14:28,531 submission_checker.py:3500 INFO] Closed Results=0, Closed Power Results=0
_
[2025-09-10 10:14:28,531 submission_checker.py:3505 INFO] Open Results=1, Open Power Results=0
_
[2025-09-10 10:14:28,531 submission_checker.py:3510 INFO] Network Results=0, Network Power Results=0
_
[2025-09-10 10:14:28,531 submission_checker.py:3515 INFO] ---
[2025-09-10 10:14:28,531 submission_checker.py:3517 INFO] Systems=1, Power Systems=0
[2025-09-10 10:14:28,531 submission_checker.py:3521 INFO] Closed Systems=0, Closed Power Systems=0
[2025-09-10 10:14:28,531 submission_checker.py:3526 INFO] Open Systems=1, Open Power Systems=0
[2025-09-10 10:14:28,531 submission_checker.py:3531 INFO] Network Systems=0, Network Power Systems=0
[2025-09-10 10:14:28,531 submission_checker.py:3536 INFO] ---
[2025-09-10 10:14:28,531 submission_checker.py:3541 INFO] SUMMARY: submission looks OK
</pre></details>


##### Server only
```
axs byname submitter , full_run \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=open --scenarios,=Server
```
