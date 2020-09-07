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

    def test_main(self):

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
        pre_df = preprocess.process(pre_df)

        # split and filter
        the_df = split_filter.call(pre_df)

        # add spectrogram
        the_df = spectrogram.add_to_df(the_df)

        # adapt to model
        train_output, test_output, train_input, test_input = model_adaptor.call(the_df)

        # make model
        the_model = model.make()

        # train model
        model.train(
            the_model, train_input, train_output, batch_size=1, epochs=10, verbose=0
        )

        # test model again
        pass_rate = model.test(the_model, train_input, train_output, verbose=0)
        print(pass_rate)
