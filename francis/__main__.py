import os
from francis import io
from francis import preprocess
from francis import spectrogram
from francis import model_adaptor
from francis import split_filter
from francis import model
from francis.spinner import Spinner
from francis.default_config import DEFAULT_CONFIG
import numpy as np
import click
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disables tensorflow debugging output


@click.group()
def cli():
    pass


@click.command()
def init():
    """produces a default config needed to use francis

    most of the time doesn't need to be changed
    """

    io.save_config(DEFAULT_CONFIG)
    print("saved config as 'francis.cfg' to the current folder")


@click.command()
@click.argument("data_path")
@click.option("-d", "--data-folder", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option("-p", "--pre-process", is_flag=True)
def train(data_path, data_folder, verbose, pre_process):
    """trains the neural network

    given an audio/dataset folder given by xeno-canto python package
    or a folder of .parquet files from a previous training session
    """
    CONFIG = {}

    # try loading config
    try:
        CONFIG = {**DEFAULT_CONFIG, **io.load_config()}
    except IOError:
        print("Can't find a francis.cfg file in this directory!")
        print("Defaulting back to default config")
        print("use 'francis init' create a default config")
        CONFIG = DEFAULT_CONFIG
    except ValueError:
        print("I can't read the francis.cfg file!")
        print("Is it formatted correctly?")
        exit(1)

    # cli overrides config
    CONFIG["PREPROCESSING_ON"] = pre_process
    CONFIG["VERBOSE"] = verbose

    # load into df
    if not data_folder:
        io.convert_to_wav(data_path, delete_old=CONFIG["DELETE_CONVERTED_MP3"])
        pre_df = io.load_into_df(data_path)

        # preprocess
        the_df = preprocess.process(
            pre_df, CONFIG["SAMPLE_RATE"], CONFIG["PREPROCESSING_ON"]
        )

        # split filter
        the_df = split_filter.call(
            the_df,
            CONFIG["SAMPLE_RATE"],
            CONFIG["SAMPLE_SECONDS"],
            CONFIG["SPLIT_FILTER_TYPE"],
            CONFIG["SPLIT_FILTER_CUTOFF"],
        )

        # creating directory to put results in
        # put a copy of the config the train_test_data and the model
        results_folder = io.results_foldername()
        try:
            os.mkdir(os.getcwd() + "/" + results_folder)
        except FileExistsError:
            print("folder already exists, try running again")
            print("it shouldn't happen")
            exit(1)

        # save df
        io.save_df(f"{results_folder}/test_train_data", the_df, rows_per_file=1000, results_folder=results_folder)

    else:
        print(f"loading from {data_path}")
        the_df = io.load_df(data_path)

    # count the num of unique label entries in the df
    num_birds = the_df["label"].nunique()

    the_df = spectrogram.add_to_df(the_df)

    # adapt to model
    print("adapting model")
    train_output, test_output, train_input, test_input = model_adaptor.adapt(
        the_df, test_size=CONFIG["TRAIN_TEST_SPLIT"]
    )

    total_samples = len(the_df.index)
    training_samples = train_output.shape[0]
    testing_samples = test_output.shape[0]
    categories = train_output.shape[1]

    print(f"total samples: {total_samples} with {categories} categories")
    print(f"about to train on {training_samples} samples!")
    print(f"then test on {testing_samples} samples!")
    # make model
    the_model = model.make(num_birds)

    if CONFIG["VERBOSE"]:
        the_model.summary()

    # train model
    print("training model")
    model.train(
        the_model,
        train_input,
        train_output,
        batch_size=CONFIG["BATCH_SIZE"],
        epochs=CONFIG["EPOCHS"],
        verbose=bool_to_int(CONFIG["VERBOSE"]),
    )

    # test model
    print("testing model")
    with Spinner():
        pass_rate = model.test(
            the_model, test_input, test_output, verbose=bool_to_int(CONFIG["VERBOSE"])
        )
    print(f"Accuracy: {pass_rate[1] * 100} %")

    # save model
    print(f"saving model to {results_folder}/model.h5")
    the_model.save(f"{results_folder}/model.h5")
    io.save_categories(f"{results_folder}/model.h5", the_df)
    io.save_config(CONFIG, f"{results_folder}/francis.cfg")

    print("✨Done!✨")


@click.command()
@click.argument("audio_sample")
def listen(audio_sample):
    """The program will try to recognise the bird in the audio sample.

    The clip must be longer than 5s.
    """

    # read into dataframe
    print("loading file into dataframe")
    pre_df = io.load_file_into_df(audio_sample)

    # preprocess
    the_df = preprocess.process(pre_df)

    # add spectrogram
    the_df = spectrogram.add_to_df(the_df)

    # adapting spectrograms
    spectrograms = model_adaptor.adapt_spectrograms(the_df)

    # load model
    print("loading model")
    with Spinner():
        the_model = load_model("model.h5")

    print("predicting ...")
    predictions = np.around(the_model.predict(spectrograms))

    print("load categories")
    categories = io.load_categories("model.h5")

    print("predictions ..")
    #  print(model_adaptor.adapt_predictions(predictions, categories))
    print(model_adaptor.summarize_predictions(predictions, categories))


def bool_to_int(boolean: bool) -> int:
    if boolean:
        return 1
    return 0


cli.add_command(init)
cli.add_command(train)
cli.add_command(listen)

if __name__ == "__main__":
    cli()
