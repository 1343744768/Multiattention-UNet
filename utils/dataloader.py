import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input
import random

class Rotate():
    def __init__(self, limit=90, prob=0.7):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)
            angle = random.uniform(-self.limit, self.limit)
            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        return img, mask


class Cutout():
    def __init__(self, max_h_size=20, max_w_size=20, fill_value=255, prob=0.5):
        self.num_holes = random.randint(10, 20)
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)
            h = img.shape[0]
            w = img.shape[1]
            # c = img.shape[2]
            # img2 = np.ones([h, w], np.float32)
            for _ in range(self.num_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(max(0, y - self.max_h_size // 2), 0, h)
                y2 = np.clip(max(0, y + self.max_h_size // 2), 0, h)
                x1 = np.clip(max(0, x - self.max_w_size // 2), 0, w)
                x2 = np.clip(max(0, x + self.max_w_size // 2), 0, w)
                img[y1: y2, x1: x2, :] = self.fill_value
                if mask is not None:
                    mask[y1: y2, x1: x2, :] = 0

            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        return img, mask


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))

        jpg, png = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw,nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        # resize image
        rand_jit1 = self.rand(1-jitter,1+jitter)
        rand_jit2 = self.rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)

        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)

        # image, label = Cutout()(image, label)  # If you want to use rotation or cutout for data enhancement, delete #
        # image, label = Rotate()(image, label)

        flip = self.rand() < .5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels
