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
        print(pre_df)

        # TODO add preprocess stage

        # split and filter
        the_df = split_filter.call(pre_df)
        print(the_df)

        # add spectrogram
        the_df = spectrogram.add_to_df(the_df)
        print(the_df)

        # adapt to model
        train_input, test_input, train_output, test_output = model_adaptor.call(the_df)

        print(f"train_input: {len(train_input)}")
        print(f"test_input: {len(test_input)}")
        print(f"train_output: {len(train_output)}")
        print(f"test_output: {len(test_output)}")
        
        # train model
        #the_model = model.make(train_input, train_output)
        
        #pass_rate = model.test(the_model, train_input, train_output, verbose=1)
        #print(pass_rate)


