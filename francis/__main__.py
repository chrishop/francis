from francis import io
from francis import preprocess
from francis import spectrogram
from francis import model_adaptor
from francis import model
import pandas as pd
import numpy as np
import click
from keras.models import load_model


@click.group()
def cli():
    pass


@click.command()
@click.argument("data_path")
def train(data_path):
    """trains the neural network

    given an audio/dataset folder given by xeno-canto python package
    or a .parquet file from a previous training session
    """

    # load into df
    if not is_file(data_path):
        io.convert_to_wav(data_path, delete_old=True)
        pre_df = io.load_into_df(data_path)

        # preprocess
        print("preprocessing")
        the_df = preprocess.process(pre_df)

        # save df
        print("saving to .parquet file")
        the_df.to_parquet("a_df.parquet")

    else:
        print("loading from parquet file")
        the_df = pd.read_parquet(data_path)

    # count the num of unique label entries in the df
    num_birds = the_df["label"].nunique()

    print("adding spectrograms")
    the_df = spectrogram.add_to_df(the_df)

    # adapt to model
    print("adapting model")
    train_output, test_output, train_input, test_input = model_adaptor.adapt(
        the_df, test_size=0.2
    )

    samples = train_output.shape

    print(f"about to train on {samples} samples!")
    # print(f"blackbird samples: {blackbird_samples}")
    # print(f"robin samples: {robin_samples}")

    # make model
    print("making model")
    the_model = model.make(num_birds)

    the_model.summary()

    # train model
    print("training model")
    model.train(
        the_model, train_input, train_output, batch_size=32, epochs=5, verbose=1
    )

    # test model
    print("testing model")
    pass_rate = model.test(the_model, test_input, test_output, verbose=0)
    print(pass_rate[1] * 100)

    # save model
    print("Saving model")
    the_model.save("model.h5")

    print("Done!")


@click.command()
@click.argument("audio_sample")
def listen(audio_sample):
    """The program will try to recognise the bird in the audio sample.

    The clip must be longer than 5s.
    """

    # read into dataframe
    print("load into df")
    pre_df = io.load_file_into_df(audio_sample)

    # preprocess
    print("preprocess")
    the_df = preprocess.process(pre_df)

    # add spectrogram
    print("adding spectrogram")
    the_df = spectrogram.add_to_df(the_df)

    # adapting spectrograms
    spectrograms = model_adaptor.adapt_spectrograms(the_df)
    print(f"there are {spectrograms.shape} spectrograms")

    # load model
    print("loading model")
    the_model = load_model("model.h5")

    print("predicting ...")
    predictions = the_model.predict(spectrograms)

    print("these are the predictions")
    print(np.around(predictions))

    print("with uncertainty")
    print(np.around(predictions, decimals=2))


def is_file(path):
    if "." in path:
        return True
    return False


cli.add_command(train)
cli.add_command(listen)

if __name__ == "__main__":
    cli()
