import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import decimal

path_root = os.getcwd()
data_path_root = Path(path_root).parent



# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 10

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def load_csv(path):
    data = pd.read_csv(path, encoding='cp1252')
    return data


def mmw_test(d1, d2):
    st, p_value = stats.mannwhitneyu(d1, d2, alternative= 'two-sided')
    return st, p_value


def fishers_exact_test (data):
    odds_ratio, p_value = stats.fisher_exact(data, alternative='two-sided')
    return odds_ratio, p_value


def report_fishers_exact_test (sys_name):
    summary_data_path = os.path.join(data_path_root, 'data', sys_name + '_summary.csv')
    data = load_csv(summary_data_path)
    # print(data.columns)
    cols_list = ['ReleaseVersion', 'JNINativeFiles', 'JNISmelly', 'JNIInducing', 'JNIInducingSmelly']
    data_df = data[cols_list]
    print(data_df)
    report_txt =''
    header = "Release, S_B, NS_B, S_NB, NS_NB, Odds_ratio, p-value"
    report_txt = header + '\n'
    # print(header)
    for i in range(data_df.shape[0]):
        d_row = data_df.iloc[i]
        smelly_buggy = d_row['JNIInducingSmelly']
        non_smelly_buggy = d_row['JNIInducing'] - d_row['JNIInducingSmelly']
        smelly_non_buggy = d_row['JNISmelly'] - d_row['JNIInducingSmelly']
        non_smelly_non_buggy = d_row['JNINativeFiles'] - (smelly_buggy + non_smelly_buggy + smelly_non_buggy)

        test_data = [[smelly_buggy, non_smelly_buggy],
                     [smelly_non_buggy, non_smelly_non_buggy]]
        odd_r, p_val = fishers_exact_test(test_data)
        txt_line = "{}, {}, {}, {}, {}, {}, {}".format(d_row['ReleaseVersion'], smelly_buggy, non_smelly_buggy,
                                        smelly_non_buggy, non_smelly_non_buggy, odd_r, np.round(p_val, decimals=5))
        # print(txt_line)
        report_txt = report_txt + txt_line + '\n'
        # break
    return report_txt


def new_conf_test():
    a = 992
    b = 165
    c = 2260
    d= 1017
    OR = 2.71
    log_OR = np.log(OR)
    print(log_OR)
    se_log_OR = np.sqrt((1.0/a)+(1.0/b)+(1.0/c)+(1.0/d))
    print(se_log_OR)
    conf_95_end = log_OR + (1.96 * se_log_OR)
    conf_95_start = log_OR - (1.96 * se_log_OR)
    print(conf_95_start, conf_95_end)
    return


def report_fishers_exact_test_new (sys_name):
    merged_data_path = os.path.join(data_path_root, 'data', 'nativedata.csv')
    data = load_csv(merged_data_path)
    #     print(data.columns)
    cols_list = ['system', 'version', 'JNINativeFlag', 'Smelly', 'InducingFlag', 'Inducing']
    data_df = data[cols_list]
    #     sys_list = list(set(data_df['system'].values))
    #     print(sys_list)
    data_df = data_df[data_df['system'] == sys_name]
    release_list = list(set(data_df['version'].values))

    report_txt = ''
    header = "Release, S_B, NS_B, S_NB, NS_NB, Odds_ratio, p-value"
    report_txt = header + '\n'
    print('Fisher\'s Test results for {}'.format(sys_name))
    print('==================================')
    print(header)
    for rl in release_list:
        # rl = 'rocksdb-5.0.2'
        rel_df = data_df[data_df['version'] == rl]  # for selected release
        #         jni_df = rel_df[rel_df['JNINativeFlag']==1] # for JNI files in selected release
        jni_df = rel_df
        smelly_buggy = jni_df[(jni_df['Smelly'] == 0) & (jni_df['InducingFlag']) == 1].shape[0]
        non_smelly_buggy = jni_df[(jni_df['Smelly'] == 0) & (jni_df['InducingFlag']) == 1].shape[0]

        sm_df = jni_df[jni_df['Smelly'] == 1]
        smnb_df = sm_df[sm_df['InducingFlag'] == 0]
        smelly_non_buggy = smnb_df.shape[0]

        # smelly_non_buggy = jni_df[(jni_df['Smelly'] > 0) & (jni_df['InducingFlag']) == 0].shape[0]
        non_smelly_non_buggy = jni_df[(jni_df['Smelly'] == 0) & (jni_df['InducingFlag']) == 0].shape[0]

        test_data = [[smelly_buggy, non_smelly_buggy],
                     [smelly_non_buggy, non_smelly_non_buggy]]
        odd_r, p_val = fishers_exact_test(test_data)

        txt_line = "{},{},{},{},{},{},{}".format(rl, smelly_buggy, non_smelly_buggy,
                                                 smelly_non_buggy, non_smelly_non_buggy, odd_r, float_to_str(p_val))
        print(txt_line)
        report_txt = report_txt + txt_line + '\n'
        break
    return report_txt


def main():

    sys_names = ['conscrypt', 'java-smt', 'javacpp', 'jpype', 'pljava', 'realm', 'rocksdb', 'vlc-android', 'zstd-jni']
    new_conf_test()
    exit()
    report = report_fishers_exact_test_new('rocksdb')
    exit()
    # ================================ Fishers test=======================================
    for i in range(len(sys_names)):
        summary_data_path = os.path.join(data_path_root, 'data', sys_names[i] + '_summary.csv')
        report_path = os.path.join(data_path_root, 'results', sys_names[i] + '_fishers_report.csv')
        report = report_fishers_exact_test(sys_names[i])
        # print(report)
        # report_file = open(report_path,"w")
        # report_file.write(report)
        # report_file.close()
    # ================================ End Fishers test=======================================
    # d1 = [50, 32, 46, 57, 62]
    # d2 = [55, 65, 75, 85, 92]
    # st, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
    # print (st)
    # print(p)
    return


if __name__ == '__main__':
    main()