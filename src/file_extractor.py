import os
import bz2
import pickle as pkl
import re

class FileExtractor:

    @staticmethod
    def extract(directory_path, skip_files):

        if os.path.isdir(directory_path) == False:
            raise ValueError("Error File Extractor: Directory does not exist")

        if not isinstance(skip_files, list):
            raise ValueError("ERROR File Extractor: Skip files must be a list of characters or strings")

        files = os.listdir(directory_path)

        output_data = []
        for file in files:

            match = False
            for skip_key in skip_files:
                regexp = re.compile(skip_key)
                if regexp.search(file):
                    match = True
                    break

            if match == False:
                fullpath = bz2.open(os.path.join(directory_path, file), 'r')
                data = pkl.load(fullpath)
                [output_data.append(data_point) for data_point in data]
                fullpath.close()

        return output_data


if __name__ == "__main__":

    data = FileExtractor.extract(directory_path='../data_seen/', skip_files=['.json'])
    a= 1

