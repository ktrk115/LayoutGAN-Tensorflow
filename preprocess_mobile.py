import os
import json
import numpy as np


def append_child(element, elements):
    if 'children' in element.keys():
        for child in element['children']:
            elements.append(child)
            elements = append_child(child, elements)
    return elements


label2index = {
    'Text': 0,
    'Text Button': 1,
    'Toolbar': 2,
    'Image': 3,
    'Icon': 4,
}

data = []
for name in sorted(os.listdir('./semantic_annotations/')):
    if not name.endswith('.json'):
        continue

    with open('./semantic_annotations/' + name) as f:
        ann = json.load(f)

    B = ann['bounds']
    if B[0] != 0 or B[1] != 0 or B[2] < 1000:
        continue
    W, H = float(B[2]), float(B[3])

    elements = append_child(ann, [])
    elements = filter(lambda e: e['componentLabel'] in label2index.keys(),
                      elements)

    if len(elements) == 0 or 9 < len(elements):
        continue

    X = np.zeros((9, 4 + 5))
    for i, elem in enumerate(elements):
        # bbox
        x1, y1, x2, y2 = elem['bounds']
        xc = (x1 + x2) / 2.
        yc = (y1 + y2) / 2.
        width = x2 - x1
        height = y2 - y1
        bbox = [xc / W, yc / H,
                width / W, height / H]

        # cls_prob
        label = elem['componentLabel']
        index = label2index[label]
        cls_prob = [0.] * 5
        cls_prob[index] = 1.

        X[i, :4] = bbox
        X[i, 4:] = cls_prob
    data.append(X)

data = np.stack(data)
np.save('data/mobile_train', data)
