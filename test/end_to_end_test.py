import unittest
import shutil
import os
from francis import io
from francis import preprocess
from francis import split_filter
from francis import spectrogram
from francis import model_adaptor
from francis import model


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        # change to test directory
        self.root = os.getcwd()
        os.chdir("test/fixtures/load_dataset")

    def tearDown(self):
        # delete downloads
        shutil.rmtree("dataset", ignore_errors=True)
        os.chdir(self.root)

    def test_train(self):

        # this stuff breaks CI
        # downloads files and converts them to wav
        # downloaded_filepaths = io.download(
        #    ["ssp:palumbus", "gen:Columba", "rec:david m"],
        #    delete_old=True
        # )
        # print(downloaded_filepaths)

        # load files into pre_df
        pre_df = io.load_into_df(".")

        # preprocess stage
        the_df = preprocess.process(pre_df, 22050, True)

        the_df = split_filter.split(
            the_df,
        )

        # add spectrogram
        the_df = spectrogram.add_to_df(the_df)

        # count the num of unique label entries in the df
        num_birds = the_df["label"].nunique()

        # adapt to model
        train_output, test_output, train_input, test_input = model_adaptor.adapt(the_df)

        # make model
        the_model = model.make(num_birds)

        # train model
        model.train(
            the_model, train_input, train_output, batch_size=1, epochs=10, verbose=0
        )

        # test model again
        pass_rate = model.test(the_model, test_input, test_output, verbose=0)
        print(pass_rate)
