import os
import json

MME_FOLDER = os.environ.get("MME_FOLDER", "<your-mme-folder>")
print(MME_FOLDER)

def get_gt(data_path=MME_FOLDER):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                try:
                    question, answer = line.strip().split('\t')
                except:
                    print(category,line)
                GT[(category, file, question)] = answer
    return GT

def processing_output(exp_dir):
    
    exp_folder_dir = exp_dir
    results_dir = os.path.join(exp_dir, '.json')
    with open(results_dir, 'r') as file:
        data = json.load(file)
    
    if not os.path.exists(exp_folder_dir):
        os.mkdir(exp_folder_dir)
    print(f"=> making dir in {exp_folder_dir}")
    
    GT = get_gt(data_path=MME_FOLDER)

    for subdata in data:
        """
            "image": "./existence/000000438304.jpg",
            "text": "Is there a sports ball in this image? Please answer this question with one word.",
            "label": "Yes",
            "cate": "existence",
            "pred": "yes"
        """
        category, prediction = subdata['cate'], subdata['pred']
        image, prompt, label = subdata['image'], subdata['text'], subdata['label']
        txt_dir = os.path.join(exp_folder_dir, category+'.txt')
        
        """
            Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"
        """
        
        Image_Name = image.split('/')[-1]
        
        if 'Please answer this question with one word.' in prompt:
            prompt = prompt.replace('Please answer this question with one word.', '').strip()
        if 'Please answer yes or no.' not in prompt:
            prompt = prompt + ' Please answer yes or no.'
            if (category, file, prompt) not in GT:
                prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
        
        Question = prompt
        Ground_Truth_Answer = label
        Your_Response = prediction.replace('\n', '').strip()
        
        with open(txt_dir, 'a') as file:
            file.writelines(Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n")
