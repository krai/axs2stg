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
<details>
<summary>
{'FID_SCORE': 23.678760940006327, 'CLIP_SCORE': 31.79914255678654}
</summary>
<pre>
, validate_accuracy
VALID : FID_SCORE=23.678760940006327
VALID : CLIP_SCORE=31.79914255678654
</pre>
</details>

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
<details>
<summary>
Samples_per_second=18.1145
</summary>
<pre>
['VALID', 'Samples_per_second=18.1145', 'target_qps=18']
</pre>
</details>

##### Compliance

###### TEST01
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=24072 \
, get performance
```
<details>
<summary>
Samples_per_second=18.113
</summary>
<pre>
['VALID', 'Samples_per_second=18.113', 'target_qps=18']
</pre>
</details>

###### TEST04
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=24072 \
, get performance
```
<details>
<summary>
Samples_per_second=18.1761
</summary>
<pre>
['VALID', 'Samples_per_second=18.1761', 'target_qps=18']
</pre>
</details>


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
<details>
<summary>
{'FID_SCORE': 23.650476437545024, 'CLIP_SCORE': 31.780069016218185}
</summary>
<pre>
, validate_accuracy
VALID : FID_SCORE=23.650476437545024
VALID : CLIP_SCORE=31.780069016218185
</pre>
</details>

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
<details>
<summary>
Completed_samples_per_second=17.88
</summary>
<pre>
['VALID', 'target_qps=18', '99.00_percentile_latency=16.552 seconds', 'target_latency=20.000 seconds', 'latency_cutoff_ratio=0.83', 'Completed_samples_per_second=17.88']
</pre>
</details>


##### Compliance

###### TEST01
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST01,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=12036 \
, get performance
```
<details>
<summary>
Completed_samples_per_second=17.86
</summary>
<pre>
['VALID', 'target_qps=18', '99.00_percentile_latency=16.398 seconds', 'target_latency=20.000 seconds', 'latency_cutoff_ratio=0.82', 'Completed_samples_per_second=17.86']
</pre>
</details>

###### TEST04
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=18,count=12036 \
, get performance
```
<details>
<summary>
Completed_samples_per_second=17.88
</summary>
<pre>
['VALID', 'target_qps=18', '99.00_percentile_latency=15.454 seconds', 'target_latency=20.000 seconds', 'latency_cutoff_ratio=0.77', 'Completed_samples_per_second=17.88']
</pre>
</details>

### Generate submission tree

#### Closed Division (use compliance tests)
```
axs byname submitter , full_run \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=closed
```
<details><summary>ERRORs before https://github.com/mlcommons/inference/pull/2337</summary><pre>
[2025-09-11 14:09:56,666 submission_checker.py:3147 INFO] Compliance test accuracy check (deterministic mode) in closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/offline/TEST01 failed
[2025-09-11 14:09:56,666 submission_checker.py:3221 ERROR] Compliance test accuracy check (non-deterministic mode) in closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/offline/TEST01 failed
[2025-09-11 14:09:56,666 submission_checker.py:2814 ERROR] compliance dir closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/offline has issues
...
[2025-09-11 14:09:56,671 submission_checker.py:3147 INFO] Compliance test accuracy check (deterministic mode) in closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/server/TEST01 failed
[2025-09-11 14:09:56,671 submission_checker.py:3221 ERROR] Compliance test accuracy check (non-deterministic mode) in closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/server/TEST01 failed
[2025-09-11 14:09:56,671 submission_checker.py:2814 ERROR] compliance dir closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/server has issues
[2025-09-11 14:09:56,671 submission_checker.py:3396 INFO] ---
[2025-09-11 14:09:56,671 submission_checker.py:3402 INFO] ---
[2025-09-11 14:09:56,671 submission_checker.py:3405 ERROR] NoResults closed/The_Stage/results
[2025-09-11 14:09:56,671 submission_checker.py:3405 ERROR] NoResults closed/The_Stage/results/h100_x8/stable-diffusion-xl/offline
[2025-09-11 14:09:56,671 submission_checker.py:3405 ERROR] NoResults closed/The_Stage/results/h100_x8/stable-diffusion-xl/server
[2025-09-11 14:09:56,671 submission_checker.py:3491 INFO] ---
[2025-09-11 14:09:56,671 submission_checker.py:3492 INFO] Results=0, NoResults=3, Power Results=0
[2025-09-11 14:09:56,671 submission_checker.py:3499 INFO] ---
[2025-09-11 14:09:56,671 submission_checker.py:3500 INFO] Closed Results=2, Closed Power Results=0
|
[2025-09-11 14:09:56,671 submission_checker.py:3505 INFO] Open Results=0, Open Power Results=0
|
[2025-09-11 14:09:56,671 submission_checker.py:3510 INFO] Network Results=0, Network Power Results=0
|
[2025-09-11 14:09:56,671 submission_checker.py:3515 INFO] ---
[2025-09-11 14:09:56,671 submission_checker.py:3517 INFO] Systems=1, Power Systems=0
[2025-09-11 14:09:56,671 submission_checker.py:3521 INFO] Closed Systems=1, Closed Power Systems=0
[2025-09-11 14:09:56,671 submission_checker.py:3526 INFO] Open Systems=0, Open Power Systems=0
[2025-09-11 14:09:56,671 submission_checker.py:3531 INFO] Network Systems=0, Network Power Systems=0
[2025-09-11 14:09:56,671 submission_checker.py:3536 INFO] ---
[2025-09-11 14:09:56,671 submission_checker.py:3538 ERROR] SUMMARY: submission has errors
</pre></details>

<details><summary>OK after https://github.com/mlcommons/inference/pull/2337</summary><pre>
[2025-09-17 16:15:38,451 submission_checker.py:3147 INFO] Compliance test accuracy check (deterministic mode) in closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/offline/TEST01 failed
...
[2025-09-17 16:15:38,456 submission_checker.py:3147 INFO] Compliance test accuracy check (deterministic mode) in closed/The_Stage/compliance/h100_x8/stable-diffusion-xl/server/TEST01 failed
[2025-09-17 16:15:38,456 submission_checker.py:3403 INFO] ---
[2025-09-17 16:15:38,456 submission_checker.py:3407 INFO] Results closed/The_Stage/results/h100_x8/stable-diffusion-xl/offline 18.1145
[2025-09-17 16:15:38,456 submission_checker.py:3407 INFO] Results closed/The_Stage/results/h100_x8/stable-diffusion-xl/server 17.8786
[2025-09-17 16:15:38,456 submission_checker.py:3409 INFO] ---
[2025-09-17 16:15:38,456 submission_checker.py:3498 INFO] ---
[2025-09-17 16:15:38,456 submission_checker.py:3499 INFO] Results=2, NoResults=0, Power Results=0
[2025-09-17 16:15:38,456 submission_checker.py:3506 INFO] ---
[2025-09-17 16:15:38,456 submission_checker.py:3507 INFO] Closed Results=2, Closed Power Results=0
|
[2025-09-17 16:15:38,456 submission_checker.py:3512 INFO] Open Results=0, Open Power Results=0
|
[2025-09-17 16:15:38,456 submission_checker.py:3517 INFO] Network Results=0, Network Power Results=0
|
[2025-09-17 16:15:38,456 submission_checker.py:3522 INFO] ---
[2025-09-17 16:15:38,456 submission_checker.py:3524 INFO] Systems=1, Power Systems=0
[2025-09-17 16:15:38,456 submission_checker.py:3528 INFO] Closed Systems=1, Closed Power Systems=0
[2025-09-17 16:15:38,456 submission_checker.py:3533 INFO] Open Systems=0, Open Power Systems=0
[2025-09-17 16:15:38,456 submission_checker.py:3538 INFO] Network Systems=0, Network Power Systems=0
[2025-09-17 16:15:38,456 submission_checker.py:3543 INFO] ---
[2025-09-17 16:15:38,456 submission_checker.py:3548 INFO] SUMMARY: submission looks OK
</pre></details>

#### Open Division (skip compliance tests)
```
axs byname submitter , full_run --submission_entry_name=laid_out_open \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=open
```
<details><summary>OK</summary><pre>
[2025-09-11 14:15:22,996 submission_checker.py:3396 INFO] ---
[2025-09-11 14:15:22,996 submission_checker.py:3400 INFO] Results open/The_Stage/results/h100_x8/stable-diffusion-xl/offline 18.1145
[2025-09-11 14:15:22,996 submission_checker.py:3400 INFO] Results open/The_Stage/results/h100_x8/stable-diffusion-xl/server 17.8786
[2025-09-11 14:15:22,996 submission_checker.py:3402 INFO] ---
[2025-09-11 14:15:22,996 submission_checker.py:3491 INFO] ---
[2025-09-11 14:15:22,996 submission_checker.py:3492 INFO] Results=2, NoResults=0, Power Results=0
[2025-09-11 14:15:22,996 submission_checker.py:3499 INFO] ---
[2025-09-11 14:15:22,996 submission_checker.py:3500 INFO] Closed Results=0, Closed Power Results=0
|
[2025-09-11 14:15:22,996 submission_checker.py:3505 INFO] Open Results=2, Open Power Results=0
|
[2025-09-11 14:15:22,996 submission_checker.py:3510 INFO] Network Results=0, Network Power Results=0
|
[2025-09-11 14:15:22,996 submission_checker.py:3515 INFO] ---
[2025-09-11 14:15:22,996 submission_checker.py:3517 INFO] Systems=1, Power Systems=0
[2025-09-11 14:15:22,996 submission_checker.py:3521 INFO] Closed Systems=0, Closed Power Systems=0
[2025-09-11 14:15:22,996 submission_checker.py:3526 INFO] Open Systems=1, Open Power Systems=0
[2025-09-11 14:15:22,996 submission_checker.py:3531 INFO] Network Systems=0, Network Power Systems=0
[2025-09-11 14:15:22,996 submission_checker.py:3536 INFO] ---
[2025-09-11 14:15:22,996 submission_checker.py:3541 INFO] SUMMARY: submission looks OK
</pre></details>

##### Offline only
```
axs byname submitter , full_run --submission_entry_name=laid_out_offline \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=open --scenarios,=Offline
```
<details><summary>OK</summary><pre>
[2025-09-11 14:06:20,232 submission_checker.py:3396 INFO] ---
[2025-09-11 14:06:20,232 submission_checker.py:3400 INFO] Results open/The_Stage/results/h100_x8/stable-diffusion-xl/offline 18.1145
[2025-09-11 14:06:20,232 submission_checker.py:3402 INFO] ---
[2025-09-11 14:06:20,232 submission_checker.py:3491 INFO] ---
[2025-09-11 14:06:20,232 submission_checker.py:3492 INFO] Results=1, NoResults=0, Power Results=0
[2025-09-11 14:06:20,232 submission_checker.py:3499 INFO] ---
[2025-09-11 14:06:20,232 submission_checker.py:3500 INFO] Closed Results=0, Closed Power Results=0
|
[2025-09-11 14:06:20,232 submission_checker.py:3505 INFO] Open Results=1, Open Power Results=0
|
[2025-09-11 14:06:20,232 submission_checker.py:3510 INFO] Network Results=0, Network Power Results=0
|
[2025-09-11 14:06:20,232 submission_checker.py:3515 INFO] ---
[2025-09-11 14:06:20,232 submission_checker.py:3517 INFO] Systems=1, Power Systems=0
[2025-09-11 14:06:20,232 submission_checker.py:3521 INFO] Closed Systems=0, Closed Power Systems=0
[2025-09-11 14:06:20,232 submission_checker.py:3526 INFO] Open Systems=1, Open Power Systems=0
[2025-09-11 14:06:20,232 submission_checker.py:3531 INFO] Network Systems=0, Network Power Systems=0
[2025-09-11 14:06:20,232 submission_checker.py:3536 INFO] ---
[2025-09-11 14:06:20,232 submission_checker.py:3541 INFO] SUMMARY: submission looks OK
</pre></details>

##### Server only
```
axs byname submitter , full_run --submission_entry_name=laid_out_server \
--task=text_to_image --model_name=stable-diffusion-xl --program_name=text_to_image_using_loadgen \
--framework=stageai --submitter=The_Stage --sut_name=h100_x8 --division=open --scenarios,=Server
```
<details><summary>OK</summary><pre>
[2025-09-11 13:09:34,606 submission_checker.py:3396 INFO] ---
[2025-09-11 13:09:34,606 submission_checker.py:3400 INFO] Results open/The_Stage/results/h100_x8/stable-diffusion-xl/server 17.8786
[2025-09-11 13:09:34,606 submission_checker.py:3402 INFO] ---
[2025-09-11 13:09:34,606 submission_checker.py:3491 INFO] ---
[2025-09-11 13:09:34,606 submission_checker.py:3492 INFO] Results=1, NoResults=0, Power Results=0
[2025-09-11 13:09:34,606 submission_checker.py:3499 INFO] ---
[2025-09-11 13:09:34,606 submission_checker.py:3500 INFO] Closed Results=0, Closed Power Results=0
|
[2025-09-11 13:09:34,606 submission_checker.py:3505 INFO] Open Results=1, Open Power Results=0
|
[2025-09-11 13:09:34,606 submission_checker.py:3510 INFO] Network Results=0, Network Power Results=0
|
[2025-09-11 13:09:34,606 submission_checker.py:3515 INFO] ---
[2025-09-11 13:09:34,606 submission_checker.py:3517 INFO] Systems=1, Power Systems=0
[2025-09-11 13:09:34,606 submission_checker.py:3521 INFO] Closed Systems=0, Closed Power Systems=0
[2025-09-11 13:09:34,606 submission_checker.py:3526 INFO] Open Systems=1, Open Power Systems=0
[2025-09-11 13:09:34,606 submission_checker.py:3531 INFO] Network Systems=0, Network Power Systems=0
[2025-09-11 13:09:34,606 submission_checker.py:3536 INFO] ---
[2025-09-11 13:09:34,606 submission_checker.py:3541 INFO] SUMMARY: submission looks OK
</pre></details>
