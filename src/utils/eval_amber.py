def eval_amber(data, file_dir):
    for cate in [
        'discriminative-attribute-state', 
        'discriminative-attribute-action', 
        'discriminative-attribute-number', 
        'discriminative-hallucination', 
    ]:
        eval_data = [item for item in data if item['cate'] == cate]
        acc, precision, recall, f1, yes, unknown = cal_results_of_amber(eval_data)
        with open(f'{file_dir}.txt', 'a') as file:
            file.write(f'{cate}: \nacc:{acc}, \nprecision:{precision}, \nrecall:{recall}, \nf1:{f1}, \nyes:{yes}, \nunknown:{unknown}')
        file.writelines("="*75+"\n")
        file.close()
        
    eval_data = [item for item in data if 'relation' in item['cate']]
    acc, precision, recall, f1, yes, unknown = cal_results_of_amber(eval_data)
    with open(f'{file_dir}.txt', 'a') as file:
        file.write(f'{cate}: \nacc:{acc}, \nprecision:{precision}, \nrecall:{recall}, \nf1:{f1}, \nyes:{yes}, \nunknown:{unknown}')
    file.writelines("="*75+"\n")
    file.close()
    

def cal_results_of_amber(data):
    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(data)
    yes_answers = 0

    # compare answers
    for index, line in enumerate(data):
        gt_answer = line["label"]
        gen_answer = line["pred"]
        # convert to lowercase
        gt_answer = gt_answer.lower()
        gen_answer = gen_answer.lower()
        # strip
        gt_answer = gt_answer.strip()
        gen_answer = gen_answer.strip()
        # pos = 'yes', neg = 'no'
        if gt_answer == 'yes':
            if 'yes' in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == 'no':
            if 'no' in gen_answer:
                true_neg += 1
            else:
                yes_answers += 1
                false_pos += 1
        else:
            print(f'Warning: unknown gt_answer: {gt_answer}')
            unknown += 1
    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    precision = true_pos / (true_pos + false_pos+1e-5)
    recall = true_pos / (true_pos + false_neg +1e-5)
    f1 = 2 * precision * recall / (precision + recall+1e-5)
    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions
    # report results
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'yes: {yes_proportion}')
    print(f'unknow: {unknown_prop}')
    
    return accuracy, precision, recall, f1, yes_proportion, unknown_prop