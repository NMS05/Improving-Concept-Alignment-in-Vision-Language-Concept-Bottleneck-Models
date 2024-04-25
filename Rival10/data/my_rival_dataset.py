import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from .original_rival_dataset import LocalRIVAL10


class Rival_Dataset(Dataset):
    def __init__(self, split, percentage_of_concept_labels_for_training):

        if split == "train":
          self.dataset = LocalRIVAL10(train=True, masks_dict=False)
        elif split == "test":
          self.dataset = LocalRIVAL10(train=False, masks_dict=False)

        # rival10 dataset class already performs: image -> tensor -> random_resized_crop(train split only) -> random_horizontal_flip(train split only) -> resize to (224,224)
        # we only normalize the image tensors here
        self.transform = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # use concept/attribute labels for X % of the samples
        self.concept_labelled_samples = []
        for i in np.random.choice(np.arange(len(self.dataset)),size=int(percentage_of_concept_labels_for_training * len(self.dataset)),replace=False): self.concept_labelled_samples.append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        (img, attr_labels, cls_name, cls_label) = (
           self.dataset[idx]['img'], 
           self.dataset[idx]['attr_labels'], 
           self.dataset[idx]['og_class_name'], 
           self.dataset[idx]['og_class_label']
        )

        img = self.transform(img)
        cls_label = torch.tensor(cls_label)
        concept_label = attr_labels

        if idx in self.concept_labelled_samples:
            use_concept_labels = torch.tensor(1)
        else:
            use_concept_labels = torch.tensor(0)

        return img, cls_label, concept_label, use_concept_labels
        # Note that the concept labels are used to estimate accuracy and not for training
    
    
# train_dataset = SS_Rival_Dataset(split="train")
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# for A,B,C,D in train_loader:
#     print(A.shape,B.shape,C.shape,D.shape)
#     break



###############################################################################################################################
###############################################################################################################################


class CSS_Rival_Dataset(Dataset):
    def __init__(self, split, true_batch_size, percentage_of_concept_labels_for_training):

        # every call returns a batch of image pairs (strictly set batch_size=1 in DataLoader of train script)
        # this makes sure that every pair in a batch belongs to a different class (to satisfy contrastive loss)
        self.batch_size = true_batch_size 
        # sort all samples in a classwise fashion
        self.classwise_samples = []
        for _ in range(10): self.classwise_samples.append([])

        if split == "train":
          self.dataset = LocalRIVAL10(train=True, masks_dict=False)
          self._parse_train_test(self.dataset)
        elif split == "test":
          self.dataset = LocalRIVAL10(train=False, masks_dict=False)
          self._parse_train_test(self.dataset)

        # rival10 dataset class already performs: image -> tensor -> random_resized_crop(train split only) -> random_horizontal_flip(train split only) -> resize to (224,224)
        # we only normalize the image tensors here
        self.transform = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # use concept/attribute labels for X % of samples
        self.concept_labelled_samples = []
        for i in np.random.choice(np.arange(len(self.dataset)),size=int(percentage_of_concept_labels_for_training * len(self.dataset)),replace=False): self.concept_labelled_samples.append(i)

    # sort all samples in a classwise fashion
    def _parse_train_test(self, _dataset):
        for i in range(len(_dataset)):
            cls_label = _dataset[i]['og_class_label']
            self.classwise_samples[cls_label].append(i)
    
    def __len__(self):
        return int(len(self.dataset)/(self.batch_size*2))

    def __getitem__(self, dummy_idx):
        
        # first choose unique classes of size = batch_size
        classes = np.random.choice(np.arange(10),size=self.batch_size,replace=False)
        # will take the shape [batch_size,2_imgs,3,224,224]
        image_pairs = []
        # will take the shape [batch_size,2_identical_labels]
        label_pairs = []
        # will take the shape [batch_size,2,18]
        concept_label_pairs = []
        # will take the shape [batch_size,2 bool values]
        use_concept_labels = []

        for i,cls_num in enumerate(classes):
            image_pairs.append([])
            label_pairs.append([])
            concept_label_pairs.append([])
            use_concept_labels.append([])

            # choose two random samples from same class
            sample_index = np.random.choice(np.asarray(self.classwise_samples[cls_num]), size=2, replace=False)

            for idx in sample_index:

                (img, attr_labels, cls_label) = (
                        self.dataset[idx]['img'], 
                        self.dataset[idx]['attr_labels'], 
                        self.dataset[idx]['og_class_label']
                        )

                image_pairs[i].append(self.transform(img))

                if cls_num != cls_label:
                    print(cls_num,cls_label)
                    assert cls_num == cls_label
                label_pairs[i].append(torch.tensor(cls_num))

                attr_labels = attr_labels.type(torch.float32)
                concept_label_pairs[i].append(attr_labels)
        
                if idx in self.concept_labelled_samples:
                    use_concept_labels[i].append(torch.tensor(1.))
                else:
                    use_concept_labels[i].append(torch.tensor(0.))
            
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
    

# dataset = CSS_Rival_Dataset(split="test", true_batch_size=8)
# print(len(dataset))
# A,B,C,D = dataset[0]
# print(A.shape,B.shape,C.shape,D.shape)
# # shapes = torch.Size([8, 2, 3, 224, 224]) torch.Size([8, 2]) torch.Size([8, 2, 18]) torch.Size([8, 2])