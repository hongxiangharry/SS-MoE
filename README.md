## Synopsis

This Github repository includes the source code for "Generalised Super Resolution for Quantitative MRI Using Self-supervised Mixture of Experts" published in MICCAI 2021; see the paper in the [link](https://link.springer.com/chapter/10.1007/978-3-030-87231-1_5). We incorporate a novel self-supervised mixture of experts (SS-MoE) framework with a DNN based multi-modal/multi-contrast super resolution, aiming to improve the robustness arising from large variance of voxel-wise intensities. SS-MoE was validated on diffusion-relaxometry MRI data from the [Super MUDI dataset](https://www.developingbrain.co.uk/data/). The software was built upon [TensorFlow 2.0](https://www.tensorflow.org/install). 



## Getting Started

### Prerequisites

The following items list the required softwares and hardwares:
* `Miniconda`: please download the installer via https://docs.conda.io/en/latest/miniconda.html for the corresponding OS and Python version.
* Nvidia GPUs: at least 12 GB memory supporting CUDA 10.1 and Tensorflow 2.3.0 .
* Operating System: This software was run and verified on CentOS 7.

### Installation

1. Clone the repo to` <PATH/TO/PROJECT/DIRECTORY>` that can be customed:
   ```sh
   cd <PATH/TO/PROJECT/DIRECTORY>
   git clone https://github.com/hongxiangharry/SS-MoE.git
   ```
2. Install SS-MoE within a Miniconda environment:
   ```sh
   cd <PATH/TO/PROJECT/DIRECTORY>
   conda config --add channels conda-forge
   conda env create -f SS-MoE-miccai/environment.yaml
   source activate ssmoe
   pip install -r SS-MoE-miccai/requirements.txt
   ```

### Dataset and models

* Data: 
  Please follow the instruction [here](https://www.developingbrain.co.uk/data/) to access the Super MUDI dataset by filling the interest form and open access agreement for data providers. Please cite and acknowledge relevant publications and funding sources therein, if your own tool/algorithm is built upon the dataset.

* Model:
  Your could train your own models, including baseline models for Stage 1 and SS-MoE, by executing `script/ssmoe-iso.sh` etc once you have Super MUDI dataset. Alternatively, the trained models could considerably be open to public via special request after the extention of the full paper.


## Usage

1. Fill in the following directories in `main/config_*.py` and `script/*.sh` to complete the entire workflow of the SS-MoE software:
   * `<PATH/TO/OUTPUT/RESULTS>`: output directory that contains the visualisation and performance.
   * `<PROCESSED_SUPER_MUDI_DATA_DIR>`: the input directory where the processed volumes are padded to the same shape.
   * `<SOURCE_SUPER_MUDI_DATA_DIR>`: the input directory that has original SUPER MUDI data.
   * `<TARGET_DIR_TO_UNZIP>`: the directory where we unzip all training patches.
   * `<SOURCE_ZIP_FILE_DIR>`: the directory that will output the zipped training patches.
2. Run `sh script/ssmoe-iso.sh` for isotropic super resolution task using SS-MoE.
3. Run `sh script/moe-iso.sh` for the same task using MoE baseline.



## Citation

Usage of the SS-MoE code requires to cite the following paper. The publications related to Super MUDI dataset should be cited and acknowledged if it is involved in your research.

```
@inproceedings{Lin2021Generalised,
	address = {Cham},
	author = {Lin, Hongxiang and Zhou, Yukun and Slator, Paddy J. and Alexander, Daniel C.},
	booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
	editor = {de Bruijne, Marleen and Cattin, Philippe C. and Cotin, St{\'e}phane and Padoy, Nicolas and Speidel, Stefanie and Zheng, Yefeng and Essert, Caroline},
	isbn = {978-3-030-87231-1},
	pages = {44--54},
	publisher = {Springer International Publishing},
	title = {Generalised Super Resolution for Quantitative MRI Using Self-supervised Mixture of Experts},
	year = {2021}}
```

