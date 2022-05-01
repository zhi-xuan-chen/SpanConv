# SpanConv: A New Convolution via Spanning Kernel Space for Lightweight Pansharpening

**Homepage:**  
https://liangjiandeng.github.io/

https://chengjin.netlify.app/

https://tianjingzhang.github.io/

- Code for paper: " SpanConv: A New Convolution via Spanning Kernel Space for Lightweight."

This is the description of how to run our training code and testing code. 

## Codes Content Structure

In order to be able to quickly get started with our code, you'd better manage your files as our standard codes content structure following.

```
codes
├── Results
│   ├── 12_reduced_wv2_512x512
│   ├── 130_reduced_wv3_128x128
│   └── 64_full_wv3_256x256
├── Weights
│   └── wv3
├── data_wv3.py
├── how-to-run.md
├── main_train_wv3.py
├── model_wv3.py
├── test_data
│   ├── 12_reduced_wv2_512x512.h5
│   ├── 130_reduced_wv3_128x128.h5
│   └── 64_full_wv3_256x256.h5
├── test_full_wv3.py
├── test_reduced_wv2.py
├── test_reduced_wv3.py
└── training_data
     ├── train_wv3_10000.h5
     └── valid_wv3_10000.h5
```

Our training data and test data are all `.h5` format, you may need to rename them for fitting our content after downloading them. Some missing empty folders need to be rebuild.

Now, let we introduce the function of each files you might use.

- data_wv3.py

  This a file to prepare the train and valid data for training. And it has used in main_train_wv3.py.  If you are not interested about it, just ignore it.

- model_wv3.py

  This file describes our SpanConv module and LightNet network in PyTorch, you can get the full details of our method implementation.

- main_train_wv3.py

  This file is a our main training file, you can try it to train our proposed LightNet with SpanConv            in WorldView-3 datasets or you just use our pretrained weights in test files. If you want to use the other datasets, make sure your dataset structure meets the requirements in `Data preparation`.

- test_full_wv3.py

  This is a test file for `64_full_wv3_256x256.h5` dataset.

- test_reduced_wv2.py

  This is a test file for `12_reduced_wv2_512x512.h5` dataset.

- test_reduced_wv3.py

  This is a test file for `130_reduced_wv3_128x128.h5` dataset.

## Data preparation

This code support the datasets in `.h5` format, and datasets for training and testing should contains the following structure respectively.

#### For training

```
YourDataset.h5
|--ms: original multispectral images in .h5 format, basically have the size of N*C*H*W 
|--lms: interpolated multispectral images in .h5 format, basically have the size of N*C*H*W 
|--pan: original panchromatic images in .h5 format, basically have the size of N*C*H*W 
|--gt: simulated ground truth images in .h5 format, basically have the size of N*C*H*W  
```

#### For testing

- 12_reduced_wv2_512x512

  ```
  
  ```

- 130_reduced_wv3_128x128

- 64_full_wv3_256x256

## Training instructions

In `main_train_wv3.py`:

If your codes content structure and datasets as same as ours, you needn't do any change luckily, just run it. 

But that's also be OK if you are a little different from ours. Some changes as following may help you.

- Modify the train/validation data input in .h5 format
- Change the pre-defined parameters and hyper parameters in `Pre-Define` and `HYPER PARAMS(Pre-Defined)`.
- Adjust the file path in `Main Function (Run first)`.

In `model_wv3.py`:

- Modify the number of navigated kernels in class `SpanConv` .
- Change input and output channel number of  class `LightNet` according to datasets.

## Testing instructions

In `model_wv3.py`:

- Modify the number of navigated kernels in class `SpanConv` .
- Change input and output channel number of  class `LightNet` according to datasets.

In `test_full_wv3.py`, `test_reduced_wv2.py` and `test_reduced_wv3.py`:

- The only place you may need to change is the `testdata` and `Results` file path if your datasets meet the requirements in `Data preparation`.
- You will get the image results in `.mat` format.
