# Francis
## Audio Recognition Neural Network 🐦

## Setup

### Linux
Install OS dependencies
```
sudo apt-get install libsndfile1
```

clone repo
```
git clone git@github.com:chrishop/francis.git
```

install package
```
chmod +x install.sh
./install
```

### Mac

Install OS dependencies
```
brew install libsndfile
```

Clone repo
```
git clone git@github.com:chrishop/francis.git
```

Install package
```
chmod +x install.sh
./install
```

## Quick Start

```
#init config
francis init

#train
francis train path/to/audio/dataset

#listen
francis listen path/to/wav/file path/to/keras/model.h5
```

## (Not So) Quick Start

### Data
First you need some data, you can collect it from your own sources or use the [xeno-canto](https://github.com/ntivirikin/xeno-canto-py) python api.

The file structure of your dataset must look like this:
```
data_folder
├── CommonBlackbird
│   ├── 434653.wav
│   └── 434654.wav
└── EuropeanRobin
    ├── 118328.wav
    ├── 127078.wav
    ├── 138019.wav
    └── 92011.wav
```

The **files must be wav or mp3** format, but they can be called what you like.

The sub-folders of the dataset must be labeled with the category they contain.


### Initialize
Use `francis init` to produce `francis.cfg` that looks like this:
```
{
    "DELETE_CONVERTED_MP3": true,
    "SAMPLE_RATE": 22050,
    "PREPROCESSING_ON": false,
    "SAMPLE_SECONDS": 5,
    "SPLIT_FILTER_TYPE": "quartile",
    "SPLIT_FILTER_CUTOFF": 0.15,
    "TRAIN_TEST_SPLIT": 0.2,
    "BATCH_SIZE": 32,
    "EPOCHS": 5
}
```

When you run francis from a folder containing this it will use the configuration values in the file.

If there is no config file it will default to the values show above.

If there are some values missing from your config the default ones will be used

All command line options such as `--pre-process` will always override the `"PREPROCESSING_ON"` option in the config


### Train & Test
Using `francis train data_folder` will train the neural network to recognise the audio by their labels to a certain degree of accuracy.

The results are saved in a folder with a randomised name (e.g. ehydi_results) so that running the cli command multiples times doesn't overwrite data
in the results folder there are 3 items:
- `francis.cfg`
- `test_train_data/`
- `model.h5`

`francis.cfg` contains a copy of the config that you used to train the neural network with

`test_train_data` contains the processed audio so that is ready to be passed into the neural network. This is useful because preprocessing the raw audio can take a long time. And loading this data takes no time at all! (comparatively)

`model.h5` is the resulting keras model that is used to decipher new audio and decide which category it falls under.

Here is an example of it running:
[![asciicast](https://asciinema.org/a/HZcKBP7xU9fpZfXkJOxgHuZlu.svg)](https://asciinema.org/a/HZcKBP7xU9fpZfXkJOxgHuZlu)


### Extra Details
```
Usage: francis train [OPTIONS] DATA_PATH

  trains the neural network

  given an audio/dataset folder given by xeno-canto python package or a
  folder of .parquet files from a previous training session

Options:
  -d, --data-folder
  -v, --verbose
  -p, --pre-process
  --help             Show this message and exit.
```

When re-running a neural network training use `--data-folder` option followed by the folder `xxxxx_results/test_train_data` this means you don't have to process the raw audio again before training your neural network! Great!

Some datasets require some scrubbing before they can be used, using the `--pre-processing` option means a high pass filter at 15000Hz and a noise reduction algorithm will be applied. We found this to not help much for audio from xeno-canto so its off by default, but your mileage may vary.

### Listen

You can now use a trained model to listen to the audio files, to decipher what you are listening to.

below is an example of usage:
[![asciicast](https://asciinema.org/a/kroCpt9oG2cA2fMjpcITSLCSL.svg)](https://asciinema.org/a/kroCpt9oG2cA2fMjpcITSLCSL)

for listening currently the audiofile **must be a wav**.



## Platforms
I've only tested this on mac and linux, but you're smart I'm sure you could get this working on windows.

