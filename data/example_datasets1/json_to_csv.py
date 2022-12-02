import json
import pandas as pd

if __name__ == '__main__':
    data = json.load(open('train_data.json', encoding='utf-8'))
    data_list = []
    for sentence in data:
        tokens = list(sentence['text'])
        labels = len(tokens) * ['O']
        if sentence['entities']:
            for entity in sentence['entities']:
                try:
                    start_idx = entity['start_idx']
                    end_idx = entity['end_idx']
                    entity_type = entity['type']
                    labels[start_idx] = 'B-' + entity_type
                    for i in range(start_idx + 1, end_idx + 1):
                        labels[i] = 'I-' + entity_type
                except IndexError:
                    continue
        for item in zip(tokens, labels):
            data_list.append((item[0], item[1]))
        data_list.append('')
    data_list = data_list[:-1]
    df = pd.DataFrame(data_list, columns=['token', 'label'])

