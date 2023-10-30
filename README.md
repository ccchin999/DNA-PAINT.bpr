# DNA-PAINT.bpr
Codes and pre-trained models for DNA-PAINT reconstruction accelerated by U-Net.

## Install first

- Nvidia CUDA
- Python 3.8
- Python Packages:
  - picassosr
  - numpy
  - PIL
  - skimage
  - xlwt
  - torch
- R
- R packages:
  - EBImage
  - hdf5r
  - configr
- Fiji
  - ImageJ plugin raw-yaml exporter (Download https://github.com/jungmannlab/imagej-raw-yaml-export/blob/master/jar/Raw_Yaml_Export.jar and put .jar file into `Fiji.app/plugins` directory)

## Usage

- Fit & localize DNA-PAINT raw data with [Picasso software](https://github.com/jungmannlab/picasso), save hdf5 files.
- Render 16-bit images in TIFF format: `Rscript /path/to/our/script/PicassoOfFrames.R /path/to/last/hdf5/file /target/path/for/reconstructed/image`
- Reconstruct super-resolution images without wide-field image: `python3 /path/to/our/script/test.py /path/to/model/file /path/to/insufficient/tiff/file /path/to/output/tiff/file`
- Reconstruct super-resolution images with wide-field image: `python3 /path/to/our/script/testWF.py /path/to/model/file /path/to/insufficient/tiff/file /path/to/output/tiff/file /path/to/widefield/file`

## Personal model training

- Prepare dataset and arrange it by: `Rscript /path/to/our/script/Roiselectcut.R /path/to/prepared/image/pair /path/to/cropped/image/dataset target-image-size`, typically `target-image-size` is `256` or `512`
- Prepare simulated micro-tubule dataset if necessary: `Rscript /path/to/our/script/mt-simulate.R structure-dense sparsity dataset-size /target/path/for/simulated/images`, `structure-dense` > 0, 1 > `sparsity` > 0
- Modify model training python script train.py or trainWF.py (optional widefield images required) and provide training dataset path (`train_data_path0`), pre-trained model path (`model_load`), number of training epochs (`epoch_num`) (typically between 200 and 2,000) and path for model saving(`model_save`). Run the script.
- Calculate PSNR, RMSE and SSIM of testing dataset: `python3 /path/to/our/script/performance.py /path/to/ground-truth/image /path/to/output/image`
