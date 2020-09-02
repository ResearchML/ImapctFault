import os
import pandas as pd
from pathlib import  Path

path_root = os.getcwd()
data_path_root = Path(path_root).parent


def load_csv(path):
    data = pd.read_csv(path, encoding='cp1252')
    return data


def clean_csv_details(source_path, target_path, text_begins_with):
    target_text =''
    cnt = 0
    line_added = True
    with open(source_path,"r") as source_file:
        for cnt, line in enumerate(source_file):
            if cnt == 0:
                target_text = line.strip()
            else:
                if line.startswith(text_begins_with):
                    target_text = target_text + '\n'
                    line_text = line.strip()
                    last_comma_index = line_text.rfind(",,")
                    line_text = line_text [:last_comma_index+2]
                    target_text = target_text + line_text
    source_file.close()

    target_file = open(target_path, "w")
    target_file.write(target_text)
    target_file.close()
    return


def main ():
    sys_names = ['conscrypt', 'java-smt','javacpp', 'jpype', 'pljava', 'realm', 'rocksdb', 'vlc-android', 'zstd-jni']
    lines_start_with = ["google/", "sosy-lab/", "bytedeco/", "jpype-project/", "tada/", "realm/", "facebook/", "videolan/", "luben/"]
    # for cleaning
    # for i in range(len(sys_names)):
    #     detail_data_path = os.path.join(data_path_root, 'data', sys_names[i]+'_details.csv')
    #     detail_cleaned_data_path = os.path.join(data_path_root, 'data', sys_names[i]+'_details_cleaned.csv')
    #     clean_csv_details(detail_data_path,detail_cleaned_data_path,lines_start_with[i])

    # for loading test after cleaning
    for i in range(len(sys_names)):
        detail_cleaned_data_path = os.path.join(data_path_root, 'data', sys_names[i]+'_details_cleaned.csv')
        data = load_csv(detail_cleaned_data_path)
        print("Rows in {} details: {}".format(sys_names[i], data.shape[0]))
    return


if __name__ == '__main__':
    main()