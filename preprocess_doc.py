import numpy as np
from pycocotools.coco import COCO
from IPython import embed

# >>> coco.loadCats(coco.getCatIds())
# [{u'id': 1, u'name': u'text', u'supercategory': u''},
#  {u'id': 2, u'name': u'title', u'supercategory': u''},
#  {u'id': 3, u'name': u'list', u'supercategory': u''},
#  {u'id': 4, u'name': u'table', u'supercategory': u''},
#  {u'id': 5, u'name': u'figure', u'supercategory': u''}]

splits = ['train', 'val']
for split in splits:
    data = []
    total, count = 0, 0
    coco = COCO('./publaynet/' + split + '.json')
    for img_id in coco.getImgIds():
        total += 1
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        if 9 < len(ann_ids):
            continue

        count += 1
        W = float(coco.loadImgs(img_id)[0]['width'])
        H = float(coco.loadImgs(img_id)[0]['height'])

        X = np.zeros((9, 4 + 5))
        for i, ann in enumerate(coco.loadAnns(ann_ids)):
            # bbox
            x1, y1, width, height = ann['bbox']
            xc = (x1 + width) / 2
            yc = (y1 + height) / 2
            bbox = [xc / W, yc / H,
                    width / W, height / H]

            # cls_prob
            index = ann['category_id'] - 1
            cls_prob = [0.] * 5
            cls_prob[index] = 1.

            X[i, :4] = bbox
            X[i, 4:] = cls_prob
        data.append(X)

    data = np.stack(data)
    np.save('data/doc_' + split, data)
    