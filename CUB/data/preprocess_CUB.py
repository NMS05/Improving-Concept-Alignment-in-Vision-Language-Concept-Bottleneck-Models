"""
This snippet processes the CUB dataset to generate train/test .txt files with the following content

<img_path>,<class_labels>,<top_x>,<top_y>,<btm_x>,<btm_y>,<312 (processed)attribute labels>

"""

def split_ids_and_contents(f_name, bboxes=False):
  f = open(f_name)
  ids = []
  contents = []
  for line in f.readlines():

    # required for cropping images
    if bboxes:
      id,x,y,w,h = line.strip().split()
      top_x = int(float(x))
      btm_x = int(float(x) + float(w))
      top_y = int(float(y))
      btm_y = int(float(y) + float(h))
      ids.append(id)
      contents.append((top_x,top_y,btm_x,btm_y))

    # for image_paths and class labels (as integers)
    else:
      id,cont = line.strip().split()
      ids.append(id)
      contents.append(cont)

  return ids, contents


image_ids, img_paths = split_ids_and_contents('CUB_200_2011/images.txt')
cls_ids, cls_labels = split_ids_and_contents('CUB_200_2011/image_class_labels.txt')
box_ids, box_coords = split_ids_and_contents('CUB_200_2011/bounding_boxes.txt', bboxes=True)
# print(image_ids[0], img_paths[0])
# print(cls_ids[0], cls_labels[0])
# print(box_ids[0], box_coords[0])


# <image_id> <attribute_id> <is_present> <certainty_id> <time>
image_attribute_labels = open("CUB_200_2011/attributes/image_attribute_labels.txt").readlines()
for i, line in enumerate(image_attribute_labels): image_attribute_labels[i] = line.strip() # remove '\n'
# print(len(image_attribute_labels))
certainity_mappings = {'1':{'1':'0.0', '2':'0.5', '3':'0.75', '4':'1.0'}, # based on CBM paper
                         '0':{'1':'0.0', '2':'0.5', '3':'0.25', '4':'0.0'}}


tr = open('cub_train.txt','w')
te = open('cub_test.txt','w')
train_test = open('CUB_200_2011/train_test_split.txt')


for i,line in enumerate(train_test.readlines()):
  id, is_train_sample = line.strip().split()
  content_to_write = []

  img_path_index = image_ids.index(id)
  image_path = img_paths[img_path_index]
  content_to_write.append(image_path)

  cls_index = cls_ids.index(id)
  cls_label = cls_labels[cls_index]
  content_to_write.append(cls_label)

  box_coords_index = box_ids.index(id)
  box_coord = box_coords[box_coords_index] # a tuple
  for loc in box_coord: content_to_write.append(str(loc))

  # for each image there are 312 lines of annos
  attribute_labels_per_image = image_attribute_labels[i*312:(i+1)*312]
  for line in attribute_labels_per_image: # iterates 312 times
    _, attribute_id, is_present, certainty_id = line.split()[:4]
    # value = float(is_present) * certainity_mappings[certainty_id]
    value = certainity_mappings[is_present][certainty_id]
    content_to_write.append(value)

  if is_train_sample == "1":
    tr.write(",".join(content_to_write)+"\n")
  else:
    te.write(",".join(content_to_write)+"\n")

tr.close()
te.close()



"""
This snippet creates class-averaged (ground-truth) concept scores. Required to estimate concept accuracy.
The original CUB dataset provides 312 concept annotations for every sample. 
This snippet returns an array of size 200 classes x 312 concepts
"""

import numpy as np

# <image_id> <attribute_id> <is_present> <certainty_id> <time>
image_attribute_labels = open("CUB_200_2011/attributes/image_attribute_labels.txt").readlines()
for i, line in enumerate(image_attribute_labels): image_attribute_labels[i] = line.strip() # remove '\n'

cls_ids = []
for line in open("CUB_200_2011/image_class_labels.txt").readlines(): cls_ids.append(line.strip().split()[-1])

concept_scores = []
for _ in range(200): concept_scores.append([])

for i in range(int(len(image_attribute_labels)/312)): # int(len(image_attribute_labels)/312) = 11788

  # for each image there are 312 lines of annos
  attribute_labels_per_image = image_attribute_labels[i*312:(i+1)*312]
  # print(attribute_labels_per_image)

  # extract the class label for an image
  image_id = int(attribute_labels_per_image[0].split()[0])
  class_id = int(cls_ids[image_id-1]) - 1 # note that cls label starts from 1 and not 0
  # print(image_id, class_id)

  # convert attribuate labels to numerical values
  numerical_attributes_per_image = []
  certainity_mappings = {'1':{'1':0.0, '2':0.5, '3':0.75, '4':1.0}, # based on CBM paper
                         '0':{'1':0.0, '2':0.5, '3':0.25, '4':0.0}}

  for line in attribute_labels_per_image: # iterates 312 times
    _, attribute_id, is_present, certainty_id = line.split()[:4]
    value = certainity_mappings[is_present][certainty_id]
    numerical_attributes_per_image.append(value)

  # print(len(numerical_attributes_per_image), numerical_attributes_per_image)
  concept_scores[class_id].append(numerical_attributes_per_image)


class_averaged_concept_scores = []

for i in range(200):
  cs_per_class = np.asarray(concept_scores[i])
  # print(i, ca_per_class.shape) # 60 images x 312 annos per class
  numerical_ca_per_class = np.mean(cs_per_class, axis=0)
  # print(numerical_ca_per_class.shape, np.around(numerical_ca_per_class[0:10], decimals=3)) # sanity check the values
  class_averaged_concept_scores.append(numerical_ca_per_class)

# save
class_averaged_concept_scores = np.asarray(class_averaged_concept_scores)
print(class_averaged_concept_scores.shape)
np.save('class_averaged_concept_scores.npy', class_averaged_concept_scores)