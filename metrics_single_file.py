import os
import json
from utility.parse_args import arg_parse

args = arg_parse()


def calculate_Precision(recommend_list, removed_tpl_list, top_n):
    hit = 0.0
    if len(recommend_list) == 0 or top_n <= 0:
        return hit
    k = min(top_n, len(recommend_list))
    for i in range(k):
        if recommend_list[i] in removed_tpl_list:
            hit = hit + 1
    return hit / top_n  # keep original denominator behavior


def calculate_Recall(recommend_list, removed_tpl_list, top_n):
    hit = 0.0
    if len(recommend_list) == 0 or len(removed_tpl_list) == 0 or top_n <= 0:
        return hit
    k = min(top_n, len(recommend_list))
    for i in range(k):
        if recommend_list[i] in removed_tpl_list:
            hit = hit + 1
    return hit / len(removed_tpl_list)


def calculate_AP(recommend_list, removed_tpl_list, top_n):
    cor_list = []
    if len(recommend_list) == 0 or top_n <= 0:
        return 0
    k = min(top_n, len(recommend_list))
    for i in range(k):
        if recommend_list[i] in removed_tpl_list:
            cor_list.append(1.0)
        else:
            cor_list.append(0.0)
    sum_cor_list = sum(cor_list)
    if sum_cor_list == 0:
        return 0

    summary = 0
    for i in range(k):
        t = (sum(cor_list[:i+1]) / (i+1)) * (cor_list[i])
        summary = summary + t
    return summary / sum_cor_list


def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    datas = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        datas.append(data)
    return datas


def calculate_metrics(datas, tpl_num, recnum_1, recnum_2):
    all_precision_1 = []
    all_precision_2 = []
    all_recall_1 = []
    all_recall_2 = []
    all_AP_1 = []
    all_AP_2 = []
    cov_1_tpls = []
    cov_2_tpls = []

    for data in datas:
        recommend_list = data['recommend_tpls']
        removed_tpl_list = data['removed_tpls']

        precision_1 = calculate_Precision(recommend_list, removed_tpl_list, recnum_1)
        all_precision_1.append(precision_1)

        precision_2 = calculate_Precision(recommend_list, removed_tpl_list, recnum_2)
        all_precision_2.append(precision_2)

        recall_1 = calculate_Recall(recommend_list, removed_tpl_list, recnum_1)
        all_recall_1.append(recall_1)

        recall_2 = calculate_Recall(recommend_list, removed_tpl_list, recnum_2)
        all_recall_2.append(recall_2)

        AP_1 = calculate_AP(recommend_list, removed_tpl_list, recnum_1)
        all_AP_1.append(AP_1)

        AP_2 = calculate_AP(recommend_list, removed_tpl_list, recnum_2)
        all_AP_2.append(AP_2)

        for tpl in recommend_list[:recnum_1]:
            if tpl not in cov_1_tpls:
                cov_1_tpls.append(tpl)

        for tpl in recommend_list[:recnum_2]:
            if tpl not in cov_2_tpls:
                cov_2_tpls.append(tpl)

    COV_1 = len(cov_1_tpls) / tpl_num
    COV_2 = len(cov_2_tpls) / tpl_num

    Mean_Precision_1 = sum(all_precision_1) / len(all_precision_1)
    Mean_Recall_1 = sum(all_recall_1) / len(all_recall_1)
    if (Mean_Precision_1 + Mean_Recall_1) == 0:
        Mean_F1_1 = 0
    else:
        Mean_F1_1 = (2 * Mean_Precision_1 * Mean_Recall_1) / (Mean_Precision_1 + Mean_Recall_1)
    MAP_1 = sum(all_AP_1) / len(all_AP_1)

    Mean_Precision_2 = sum(all_precision_2) / len(all_precision_2)
    Mean_Recall_2 = sum(all_recall_2) / len(all_recall_2)
    if (Mean_Precision_2 + Mean_Recall_2) == 0:
        Mean_F1_2 = 0
    else:
        Mean_F1_2 = (2 * Mean_Precision_2 * Mean_Recall_2) / (Mean_Precision_2 + Mean_Recall_2)
    MAP_2 = sum(all_AP_2) / len(all_AP_2)

    line1 = 'Top-rm  => MP: %2f  MR: %2f  MF: %2f  MAP: %2f  COV: %2f' % (
        Mean_Precision_1, Mean_Recall_1, Mean_F1_1, MAP_1, COV_1
    )
    line2 = 'Top-2rm => MP: %2f  MR: %2f  MF: %2f  MAP: %2f  COV: %2f' % (
        Mean_Precision_2, Mean_Recall_2, Mean_F1_2, MAP_2, COV_2
    )

    return line1, line2


def save_metrics(output_path, method, fold, rm_num, line1, line2):
    os.makedirs(output_path, exist_ok=True)

    # 1) Per fold-rm file
    per_file = os.path.join(output_path, f"metrics_{method}_{fold}_{rm_num}.txt")
    with open(per_file, "w", encoding="utf-8") as f:
        f.write(line1 + "\n")
        f.write(line2 + "\n")

    # 2) Append to ALL summary file
    all_file = os.path.join(output_path, f"ALL_metrics_{method}.txt")
    with open(all_file, "a", encoding="utf-8") as f:
        f.write(f"[fold={fold}, rm={rm_num}]\n")
        f.write(line1 + "\n")
        f.write(line2 + "\n\n")

    return per_file, all_file


if __name__ == '__main__':
    tpl_num = args.tpl_range
    method = 'Atten_TPL'
    fold = args.fold
    rm_num = args.rm
    RecNum_1 = rm_num
    RecNum_2 = 2 * rm_num
    test = 'testing_'
    output_path = args.output_path

    pred_file = f"{output_path}/{test}{method}_{fold}_{rm_num}.json"
    data = read_json_file(pred_file)

    line1, line2 = calculate_metrics(data, tpl_num, RecNum_1, RecNum_2)

    # print to console/log (same as before)
    print(line1)
    print(line2)

    # save to file(s)
    per_file, all_file = save_metrics(output_path, method, fold, rm_num, line1, line2)
    print(f"Saved metrics -> {per_file}")
    print(f"Appended summary -> {all_file}")