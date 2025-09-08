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

See below example commands for the Server/Offline scenarios and under the Accuracy/Performance/Compliance modes.

See actual commands used for the v5.1 submissions under the corresponding subdirectories of `closed/The_Stage/measurements/`.

#### Offline

##### Accuracy
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=<qps> \
, get accuracy_report
```

##### Performance
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=<qps>,count=<count> \
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
qps=<qps>,count=<count>
```

###### TEST04
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=<qps>,count=<count>
```

#### Server

##### Accuracy
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=AccuracyOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=<qps> \
, get accuracy_report
```

##### Performance
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=<qps>,count=<count> \
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
qps=<qps>,count=<count>
```

###### TEST04
```
axs byquery loadgen_output,task=text_to_image,framework=stageai,backend=stageai,device=cuda,dtype=fp16,\
model_path=stabilityai/stable-diffusion-xl-base-1.0,dataset=coco-1024,profile=stable-diffusion-xl-stageai,\
sut_name=h100_x8,num_gpus=8,axs_device_id=0+1+2+3+4+5+6+7,\
loadgen_scenario=Server,loadgen_mode=PerformanceOnly,loadgen_compliance_test=TEST04,\
loadgen_dataset_size=5000,loadgen_buffer_size=5000,\
qps=<qps>,count=<count>
```

### Generate submission tree

`TODO`
