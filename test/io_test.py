import unittest
import os
from francis import io
import glob


class IOTest(unittest.TestCase):
    def test_load_folder(self):

        the_df = io.load_into_df("test/fixtures/load_dataset/audio")

        # it has the correct columns
        self.assertEqual(
            the_df.columns.to_list(), ["id", "label", "audio_buffer"]
        )

        # it has an id
        self.assertEqual(the_df.iloc[0].to_list()[0], "434652")

        # it has a label
        self.assertEqual(the_df.iloc[0].to_list()[1], "CommonBlackbird")

        # it fetches from both subfolders
        self.assertEqual(
            the_df["label"].to_list(),
            ["CommonBlackbird", "CommonWoodPidgeon"],
        )

    @unittest.skip("can't get this working with ci, uncomment to test locally")
    def test_convert_to_wav(self):
        # copy mp3 files in from safe folder
        file_from = "test/fixtures/convert_dataset/434652.mp3"
        file_to = "test/fixtures/convert_dataset/audio/CommonBlackbird/"
        os.system(f"cp {file_from} {file_to}")

        file_from = "test/fixtures/convert_dataset/463432.mp3"
        file_to = "test/fixtures/convert_dataset/audio/CommonWoodPidgeon/"
        os.system(f"cp {file_from} {file_to}")

        converted = io.convert_to_wav(
            "test/fixtures/convert_dataset/audio", delete_old=True
        )

        # files from both folders have been converted
        self.assertTrue("CommonBlackbird" in converted[0])
        self.assertTrue("CommonWoodPidgeon" in converted[1])

        # the extension is .wav
        self.assertTrue(".wav" in converted[0])

        # mp3 files have been deleted
        self.assertEqual(
            glob.glob("test/fixtures/convert_dataset/audio/**/*.mp3"), []
        )

        # delete wav files
        for filepath in glob.glob(
            "test/fixtures/convert_dataset/audio/**/*.wav"
        ):
            os.remove(filepath)
