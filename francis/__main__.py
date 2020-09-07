import sys
from francis import io
from francis import preprocess
from francis import split_filter
from francis import spectrogram
from francis import model_adaptor
from francis import model
import pandas as pd


def main():
    # give folder name to process
    path = sys.argv[1]
    pre_df = ''

    # load into df
    if not is_file(path):
        io.convert_to_wav(path, delete_old=True)
        pre_df = io.load_into_df(path)
        
        # preprocess
        print("preprocessing")
        pre_df = preprocess.process(pre_df)
        
        # split and filter
        print("split filter")
        the_df = split_filter.call(pre_df)
        
        # save df
        print("saving to .parquet file")
        the_df.to_parquet('full_df.parquet')

    else:
        print("loading from parquet file")
        the_df = pd.read_parquet(path)
        
    print("adding spectrograms")
    the_df = spectrogram.add_to_df(the_df)
    
    # adapt to model
    print("adapting model")
    train_output, test_output, train_input, test_input = model_adaptor.call(the_df, test_size=0.2)

    samples = train_output.shape


    print(f"about to train on {samples} samples!")
    # print(f"blackbird samples: {blackbird_samples}")
    # print(f"robin samples: {robin_samples}")

    # make model
    print("making model")
    the_model = model.make()

    the_model.summary()

    # train model
    print("training model")
    model.train(
        the_model, train_input, train_output, batch_size=32, epochs=5, verbose=1
    )

    # test model
    print("testing model")
    pass_rate = model.test(the_model, train_input, train_output, verbose=0)
    print(pass_rate)


def is_file(path):
    if '.' in path:
        return True
    return False


if __name__ == "__main__":
    main()
