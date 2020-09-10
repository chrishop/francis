import unittest
import os
from francis import io
import glob
import os
import pandas as pd


class IOTest(unittest.TestCase):
    def test_load_folder(self):

        the_df = io.load_into_df("test/fixtures/load_dataset/audio")

        # it has the correct columns
        self.assertEqual(the_df.columns.to_list(), ["id", "label", "audio_buffer"])

        # it has an id
        self.assertEqual(the_df.iloc[0].to_list()[0], "434652")

        # it has a label
        self.assertEqual(the_df.iloc[0].to_list()[1], "CommonBlackbird")

        # it fetches from both subfolders
        self.assertEqual(
            the_df["label"].to_list(),
            ["CommonBlackbird", "CommonWoodPidgeon"],
        )

    def test_load_file(self):

        the_df = io.load_file_into_df(
            "test/fixtures/load_dataset/audio/CommonBlackbird/434652.wav"
        )

        # it has the correct columns
        self.assertEqual(the_df.columns.to_list(), ["id", "label", "audio_buffer"])

        # it only has one item
        self.assertEqual(len(the_df.index), 1)

        # it has an id
        self.assertEqual(the_df.iloc[0].to_list()[0], "434652")

        # it has a label
        self.assertEqual(the_df.iloc[0].to_list()[1], "CommonBlackbird")

    def test_save_categories(self):
        fake_df = pd.DataFrame({"label": ["CommonBlackbird", "EurasianRobin", "Wren"]})
        expected_json = '["CommonBlackbird", "EurasianRobin", "Wren"]'

        io.save_categories("test/fixtures/test_category_save.json", fake_df)

        # load json file
        with open("test/fixtures/test_category_save.json") as test_file:
            resulting_json = test_file.read()

            self.assertEqual(resulting_json, expected_json)

        os.remove("test/fixtures/test_category_save.json")
