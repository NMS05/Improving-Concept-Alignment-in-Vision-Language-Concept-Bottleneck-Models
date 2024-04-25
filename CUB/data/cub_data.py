from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np


"""
Returns image and class-label
Used to train vanilla CLIP VL-CBM
"""
class Processed_CUB_Dataset(Dataset):
    def __init__(self, split):

        if split == "train":
          self.annos = open("data/cub_train.txt").readlines()
        elif split == "test":
          self.annos = open("data/cub_test.txt").readlines()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224,224), antialias=True),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label, top_x, top_y, btm_x, btm_y = self.annos[idx].strip().split(",")[0:6]

        image = Image.open("CUB_200_2011/images/" + img_path)
        image = image.crop((int(top_x),int(top_y),int(btm_x),int(btm_y)))
        image = self.transform(image)

        # note that cls label starts from 1 and not 0
        label = int(cls_label)-1

        return image, label # to be used for training VL-CBM


# train_dataset = Processed_CUB_Dataset(split="test")
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# for images, labels in train_loader:
#     print(images.shape, labels.shape)
#     # break

# # Manually identify GrayScale images and discard them from train/test splits
# for i in range(len(train_dataset)):
#     img_path, image, label = train_dataset[i]
#     if image.shape[0] !=3: print(img_path)



# ===================================================================================================================================
# ===================================================================================================================================


"""
Returns image pairs, class-label apirs and concept label pairs
Used to train CSS-VL-CBM with Concept Proj.
"""
class CSS_CUB_Dataset(Dataset):
    def __init__(self, split, true_batch_size, percentage_of_concept_labels_for_training = 0.3):

        # every call returns a batch of image pairs (strictly set batch_size=1 in DataLoader of train script)
        # this makes sure that every pair in a batch belongs to a different class (to satisfy contrastive loss)
        self.batch_size = true_batch_size 
        # sort all samples in a classwise fashion
        self.classwise_samples = []
        for _ in range(200): self.classwise_samples.append([])
        
        if split == "train":
          self.data_len = len(open("data/cub_train.txt").readlines())
          self._parse_train_test(open("data/cub_train.txt").readlines())

        elif split == "test":
          self.data_len = len(open("data/cub_test.txt").readlines())
          self._parse_train_test(open("data/cub_test.txt").readlines())

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((224,224), antialias=True),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # use concept labels for only 30% of the samples
        self.concept_labelled_samples = []
        for i in np.random.choice(np.arange(self.data_len),size=int(percentage_of_concept_labels_for_training * self.data_len),replace=False): self.concept_labelled_samples.append(i)

    # sort all samples in a classwise fashion
    def _parse_train_test(self, anno_list):
        for i in anno_list:
            cls_label = i.strip().split(",")[1]
            self.classwise_samples[int(cls_label)-1].append(i.strip().split(","))
    
    # value of __len__ is chosen such that each sample is approximately seen only once per epoch
    def __len__(self):
        return int(self.data_len/(self.batch_size*2))

    def __getitem__(self, dummy_idx):

        # first choose unique classes of size = batch_size = 64
        classes = np.random.choice(np.arange(200),size=self.batch_size,replace=False)
        # will take the shape [batch_size,2_imgs,3,224,224]
        image_pairs = []
        # will take the shape [batch_size,2_identical_labels]
        label_pairs = []
        # will take the shape [batch_size,2,31]
        concept_label_pairs = []
        # will take the shape [batch_size,2 bool values]
        use_concept_labels = []

        for i,cls_num in enumerate(classes):
            image_pairs.append([])
            label_pairs.append([])
            concept_label_pairs.append([])
            use_concept_labels.append([])

            # choose two random samples from same class
            sample_index = np.random.choice(np.arange(len(self.classwise_samples[cls_num])), size=2, replace=False)

            for idx in sample_index:

                img_path, cls_label, top_x, top_y, btm_x, btm_y = self.classwise_samples[cls_num][idx][0:6]
                image = Image.open("CUB_200_2011/images/" + img_path)
                image = image.crop((int(top_x),int(top_y),int(btm_x),int(btm_y)))
                image = self.transform(image)
                image_pairs[i].append(image)

                label_pairs[i].append(torch.tensor(cls_num))

                concept_label = []
                for cl in self.classwise_samples[cls_num][idx][6:]: concept_label.append(float(cl))
                concept_label_pairs[i].append(torch.tensor(concept_label))
        
                if idx in self.concept_labelled_samples:
                    use_concept_labels[i].append(torch.tensor(1))
                else:
                    use_concept_labels[i].append(torch.tensor(0))
            
            # shape=(2,3,224,224)
            image_pairs[i] = torch.stack(image_pairs[i],0)
            label_pairs[i] = torch.stack(label_pairs[i],0)
            concept_label_pairs[i] = torch.stack(concept_label_pairs[i],0)
            use_concept_labels[i] = torch.stack(use_concept_labels[i],0)
        
        image_pairs = torch.stack(image_pairs,0)
        label_pairs = torch.stack(label_pairs,0)
        concept_label_pairs = torch.stack(concept_label_pairs,0)
        use_concept_labels = torch.stack(use_concept_labels,0)

        return image_pairs, label_pairs, concept_label_pairs, use_concept_labels

# train_dataset = SupCon_CUB_Dataset(split="train", true_batch_size=64)
# A,B,C,D = train_dataset[0]
# print(A.shape,B.shape,C.shape,D.shape)
# # shapes = torch.Size([64, 2, 3, 224, 224]) torch.Size([64, 2]) torch.Size([64, 2, 312]) torch.Size([64, 2])