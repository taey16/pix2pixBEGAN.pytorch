import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
  classes = ['imgA', 'imgB']
  class_to_idx = {'imgA': 0, 'imgB': 1}
  assert os.path.isdir(os.path.join(dir, classes[0]))
  assert os.path.isdir(os.path.join(dir, classes[1]))
  return classes, class_to_idx

def make_dataset(dir, classes):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(os.path.join(dir, classes[0]))):
    imageSet = []
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, classes[0], fname)
        item = path
        imageSet.append(item)
    images.append(imageSet)
  for root, _, fnames in sorted(os.walk(os.path.join(dir, classes[1]))):
    imageSet = []
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, classes[1], fname)
        item = path
        imageSet.append(item)
    images.append(imageSet)
  return images

def default_loader(path):
  return Image.open(path).convert('RGB')

class trans(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    #import pdb; pdb.set_trace()
    classes, class_to_idx = find_classes(root)
    imgs = make_dataset(root, classes)
    if len(imgs[0]) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\/imgA\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    if len(imgs[1]) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\/imgB\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    self.imgs = imgs
    self.transform = transform
    self.loader = loader
    self.len_img = []
    self.len_img.append(len(imgs[0]))
    self.len_img.append(len(imgs[1]))

    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, _):
    output = []
    for idx in range(2):
      index = np.random.randint(self.len_img[idx], size=1)[0]
      path = self.imgs[idx][index]
      output.append(self.loader(path))
    if self.transform is not None:
      imgA, imgB = self.transform(output[0], output[1])
    return imgA, imgB

  def __len__(self):
    return len(self.imgs[0])
