import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.autograd import Variable


def get_test_dataset(args):

    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domA_test = CustomDataset(os.path.join(args.root, 'testA.txt'), transform=comp_transform)
    domB_test = CustomDataset(os.path.join(args.root, 'testB.txt'), transform=comp_transform)

    return domA_test, domB_test

def get_train_dataset(args):

    comp_transform = transforms.Compose([
        transforms.CenterCrop(args.crop),
        transforms.Resize(args.resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    domA_train = CustomDataset(os.path.join(args.root, 'trainA.txt'), transform=comp_transform)
    domB_train = CustomDataset(os.path.join(args.root, 'trainB.txt'), transform=comp_transform)

    return domA_train, domB_train


def save_imgs_explainability(args, classif, e_common, e_separate_A, e_separate_B, decoder, epoch, device, A=True, correct=False, is_common=False):
    ''' saves images of reconstruction of opposite label'''
    test_domA, test_domB = get_test_imgs(args, 1000)

    list_of_imgs = []
    k = 0
    i = 0
    exps = []
    corr = ''

    if A:
        test_dom = test_domA
        label = 1
    else:
        test_dom = test_domB
        label = 0

    if correct: corr = 'correct'
    else: corr = 'wrong'

    while k <= args.num_display+1 and i < len(test_dom):
        with torch.no_grad():
            if is_common:
                common = e_common(test_dom[i].unsqueeze(0))
            else:
                common = torch.full((1, (512 - 2 * args.sep) * (args.resize // 32) * (args.resize // 32)), 0).to(device)
            separate_A = e_separate_A(test_dom[i].unsqueeze(0))
            separate_B = e_separate_B(test_dom[i].unsqueeze(0))
            encoding = torch.cat([common, separate_A, separate_B], dim=1)
            outputs = classif(encoding)
            value, predicted = outputs.max(1)

            if correct:
                if (predicted.data.tolist()[0] == label):
                    list_of_imgs += [test_dom[i]]
                    k += 1
            else:
                if (predicted.data.tolist()[0] != label and value.data.tolist()[0] > 0.65):
                    list_of_imgs += [test_dom[i]]
                    k += 1
            i += 1

    if i == len(test_dom): return

    for i in range(k):
        with torch.no_grad():
            exps.append(list_of_imgs[i].unsqueeze(0))

    zero_encoding = torch.full((1, args.sep * (args.resize // 32) * (args.resize // 32)), 0)
    if torch.cuda.is_available():
        zero_encoding = zero_encoding.cuda()

    for i in range(k):
        common = e_common(list_of_imgs[i].unsqueeze(0))
        separate_A = e_separate_A(list_of_imgs[i].unsqueeze(0))
        separate_B = e_separate_B(list_of_imgs[i].unsqueeze(0))

        A_encoding = torch.cat([common, separate_A, zero_encoding], dim=1)
        A_decoding = decoder(A_encoding).detach()
        exps.append(A_decoding)

    for i in range(k):
        common = e_common(list_of_imgs[i].unsqueeze(0))
        separate_A = e_separate_A(list_of_imgs[i].unsqueeze(0))
        separate_B = e_separate_B(list_of_imgs[i].unsqueeze(0))

        B_encoding = torch.cat([common, zero_encoding, separate_B], dim=1)
        B_decoding = decoder(B_encoding).detach()
        exps.append(B_decoding)


    with torch.no_grad():
        exps = torch.cat(exps, 0)

        if A:
            vutils.save_image(exps,
                              '%s/experiments_%06d_%d-A-%s.png' % (args.out,
                                                                   epoch,
                                                                   k,
                                                                   corr),
                              normalize=True, nrow=k)
        else:
            vutils.save_image(exps,
                              '%s/experiments_%06d_%d-B-%s.png' % (args.out,
                                                                   epoch,
                                                                   k,
                                                                corr),
                              normalize=True, nrow=k)


def save_imgs(args, e_common, e_separate_A, e_separate_B, decoder, iters, BtoA=True, num_offsets=1):
    ''' saves images of translation B -> A or A -> B'''
    test_domA, test_domB = get_test_imgs(args)

    for k in range(num_offsets):
        exps = []
        for i in range(k * args.num_display, (k + 1) * args.num_display):
            with torch.no_grad():
                if i == k * args.num_display:
                    filler = test_domB[i].unsqueeze(0).clone()
                    exps.append(filler.fill_(0))

                if BtoA:
                    exps.append(test_domB[i].unsqueeze(0))
                else:
                    exps.append(test_domA[i].unsqueeze(0))

        if BtoA:
            for i in range(k * args.num_display, (k + 1) * args.num_display):
                exps.append(test_domA[i].unsqueeze(0))
                separate_A = e_separate_A(test_domA[i].unsqueeze(0))
                for j in range(k * args.num_display, (k + 1) * args.num_display):
                    with torch.no_grad():
                        common_B = e_common(test_domB[j].unsqueeze(0))
                        zero_encoding = torch.full((1, args.sep * (args.resize
                                                                   // 32) * (args.resize // 32)), 0)
                        if torch.cuda.is_available():
                            zero_encoding = zero_encoding.cuda()

                        BA_encoding = torch.cat([common_B, separate_A, zero_encoding], dim=1)
                        BA_decoding = decoder(BA_encoding)
                        exps.append(BA_decoding)
        else:
            for i in range(k * args.num_display, (k + 1) * args.num_display):
                exps.append(test_domB[i].unsqueeze(0))
                separate_B = e_separate_B(test_domB[i].unsqueeze(0))
                for j in range(k * args.num_display, (k + 1) * args.num_display):
                    with torch.no_grad():
                        common_A = e_common(test_domA[j].unsqueeze(0))
                        zero_encoding = torch.full((1, args.sep * (args.resize
                                                                   // 32) * (args.resize // 32)), 0)
                        if torch.cuda.is_available():
                            zero_encoding = zero_encoding.cuda()

                        AB_encoding = torch.cat(
                            [common_A, zero_encoding, separate_B], dim=1)
                        AB_decoding = decoder(AB_encoding)
                        exps.append(AB_decoding)

        with torch.no_grad():
            exps = torch.cat(exps, 0)

        if BtoA:
            vutils.save_image(exps,
                              '%s/experiments_%06d_%d-BtoA.png' % (args.out,
                                                                   iters,
                                                                   k),
                              normalize=True, nrow=args.num_display + 1)
        else:
            vutils.save_image(exps,
                              '%s/experiments_%06d_%d-AtoB.png' % (args.out,
                                                                   iters,
                                                                   k),
                              normalize=True, nrow=args.num_display + 1)


def get_test_imgs(args, bs = 64):
    domA_test, domB_test = get_test_dataset(args)

    domA_test_loader = torch.utils.data.DataLoader(domA_test, batch_size=bs,
                                                   shuffle=False, num_workers=6)
    domB_test_loader = torch.utils.data.DataLoader(domB_test, batch_size=bs,
                                                   shuffle=False, num_workers=6)

    for domA_img in domA_test_loader:
        if torch.cuda.is_available():
            domA_img = domA_img.cuda()
        domA_img = domA_img.view((-1, 3, args.resize, args.resize))
        domA_img = domA_img[:]
        break

    for domB_img in domB_test_loader:
        if torch.cuda.is_available():
            domB_img = domB_img.cuda()
        domB_img = domB_img.view((-1, 3, args.resize, args.resize))
        domB_img = domB_img[:]
        break

    return domA_img, domB_img


def save_model(out_file, e_common, e_separate_A, e_separate_B, decoder, ae_opt, disc, disc_opt, iters):
    state = {
        'e_common': e_common.state_dict(),
        'e_separate_A': e_separate_A.state_dict(),
        'e_separate_B': e_separate_B.state_dict(),
        'decoder': decoder.state_dict(),
        'ae_opt': ae_opt.state_dict(),
        'disc': disc.state_dict(),
        'disc_opt': disc_opt.state_dict(),
        'iters': iters
    }
    torch.save(state, out_file)
    return


def load_model(load_path, e_common, e_separate_A, e_separate_B, decoder, ae_opt, disc, disc_opt):
    state = torch.load(load_path)
    e_common.load_state_dict(state['e_common'])
    e_separate_A.load_state_dict(state['e_separate_A'])
    e_separate_B.load_state_dict(state['e_separate_B'])
    decoder.load_state_dict(state['decoder'])
    ae_opt.load_state_dict(state['ae_opt'])
    disc.load_state_dict(state['disc'])
    disc_opt.load_state_dict(state['disc_opt'])
    return state['iters']


def load_model_for_eval(load_path, e_common, e_separate_A, e_separate_B, decoder, ):
    state = torch.load(load_path)
    e_common.load_state_dict(state['e_common'])
    e_separate_A.load_state_dict(state['e_separate_A'])
    e_separate_B.load_state_dict(state['e_separate_B'])
    decoder.load_state_dict(state['decoder'])
    return state['iters']


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def edges_loader(path, train=True):
    image = Image.open(path).convert('RGB')
    image_A = image.crop((0, 0, 256, 256))
    image_B = image.crop((0, 256, 512, 256))

    if train:
        return image_A
    else:
        return image_B


def default_loader(path):
    return Image.open(path).convert('RGB')


class Logger():
    def __init__(self, path):
        self.full_path = '%s/log.txt' % path
        self.log_file = open(self.full_path, 'w+')
        self.log_file.close()
        self.map = {}

    def add_value(self, tag, value):
        self.map[tag] = value

    def log(self, iter):
        self.log_file = open(self.full_path, 'a')
        self.log_file.write('iter: %7d' % iter)
        for k, v in self.map.items():
            self.log_file.write('\t %s: %10.7f' % (k, v))
        self.log_file.write('\n')
        self.log_file.close()

    def reset(self):
        self.map = {}


class CustomDataset(data.Dataset):
    def __init__(self, path, transform=None, return_paths=False,
                 loader=default_loader):
        super(CustomDataset, self).__init__()

        with open(path) as f:
            imgs = [s.replace('\n', '') for s in f.readlines()]

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + path + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, train=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    files = sorted(os.walk(dir))

    for root, _, fnames in files:
        random.seed(0)
        random.shuffle(fnames)

        if train is not None:
            cut_point = int(len(fnames) * 0.83)
            if train:
                fnames = fnames[:cut_point]
            else:
                fnames = fnames[cut_point:]

        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, train=None):
        imgs = make_dataset(root, train=train)  # sorted(make_dataset(root, train=train))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.train = train

    def __getitem__(self, index):

        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def save_stripped_imgs(args, e_common, e_separate_A, e_separate_B, decoder, iters, A=True):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    zero_encoding = torch.full((1, args.sep * (args.resize // 32) * (args.resize // 32)), 0)
    # zero_encoding = torch.full((1, 12, 32, 32), 0)
    if torch.cuda.is_available():
        zero_encoding = zero_encoding.cuda()

    for i in range(args.num_display):
        if A:
            image = test_domA[i]
        else:
            image = test_domB[i]
        exps.append(image.unsqueeze(0))
        common = e_common(image.unsqueeze(0))
        content_zero_encoding = torch.full(common.size(), 0)
        if torch.cuda.is_available():
            content_zero_encoding = content_zero_encoding.cuda()
        separate_A = e_separate_A(image.unsqueeze(0))
        separate_B = e_separate_B(image.unsqueeze(0))
        exps.append(decoder(torch.cat([common, zero_encoding, zero_encoding], dim=1)))
        exps.append(decoder(torch.cat([content_zero_encoding, separate_A, zero_encoding], dim=1)))
        exps.append(decoder(torch.cat([content_zero_encoding, zero_encoding, separate_B], dim=1)))

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    if A:
        vutils.save_image(exps,
                          '%s/experiments_%06d-Astripped.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display)
    else:
        vutils.save_image(exps,
                          '%s/experiments_%06d-Bstripped.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display)


def save_chosen_imgs_removal(args, e_common, decoder, iters, list, BtoA=True):
    ''' saves images of translation B -> A or A -> B'''
    test_domA, test_domB = get_test_imgs(args)

    exps = []
    for i in range(args.num_display):
        with torch.no_grad():
            if BtoA:
                exps.append(test_domB[list[i]].unsqueeze(0))
            else:
                exps.append(test_domA[list[i]].unsqueeze(0))

    if BtoA:
        for i in list:
            with torch.no_grad():
                common_B = e_common(test_domB[i].unsqueeze(0))
                zero_encoding = torch.full((1, args.sep * (args.resize
                                                           // 32) * (args.resize // 32)), 0)
                if torch.cuda.is_available():
                    zero_encoding = zero_encoding.cuda()

                BA_encoding = torch.cat([common_B, zero_encoding, zero_encoding], dim=1)
                BA_decoding = decoder(BA_encoding)
                exps.append(BA_decoding)
    else:
        for i in list:
            with torch.no_grad():
                common_A = e_common(test_domA[i].unsqueeze(0))
                zero_encoding = torch.full((1, args.sep * (args.resize
                                                           // 32) * (args.resize // 32)), 0)
                if torch.cuda.is_available():
                    zero_encoding = zero_encoding.cuda()

                AB_encoding = torch.cat(
                    [common_A, zero_encoding, zero_encoding], dim=1)
                AB_decoding = decoder(AB_encoding)
                exps.append(AB_decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    if BtoA:
        vutils.save_image(exps,
                          '%s/experiments_%06d-BtoA.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display)
    else:
        vutils.save_image(exps,
                          '%s/experiments_%06d-AtoB.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display)

def save_chosen_imgs(args, e_common, e_separate_A, e_separate_B, decoder, iters, listA, listB, BtoA=True):
    ''' saves images of translation B -> A or A -> B'''
    test_domA, test_domB = get_test_imgs(args)

    exps = []
    for i in range(args.num_display):
        with torch.no_grad():
            if i == 0:
                filler = test_domB[i].unsqueeze(0).clone()
                exps.append(filler.fill_(0))

            if BtoA:
                exps.append(test_domB[listB[i]].unsqueeze(0))
            else:
                exps.append(test_domA[listA[i]].unsqueeze(0))

    if BtoA:
        for i in listA:
            exps.append(test_domA[i].unsqueeze(0))
            separate_A = e_separate_A(test_domA[i].unsqueeze(0))
            for j in listB:
                with torch.no_grad():
                    common_B = e_common(test_domB[j].unsqueeze(0))
                    zero_encoding = torch.full((1, args.sep * (args.resize
                                                               // 32) * (args.resize // 32)), 0)
                    if torch.cuda.is_available():
                        zero_encoding = zero_encoding.cuda()

                    BA_encoding = torch.cat([common_B, separate_A, zero_encoding], dim=1)
                    BA_decoding = decoder(BA_encoding)
                    exps.append(BA_decoding)
    else:
        for i in listB:
            exps.append(test_domB[i].unsqueeze(0))
            separate_B = e_separate_B(test_domB[i].unsqueeze(0))
            for j in listA:
                with torch.no_grad():
                    common_A = e_common(test_domA[j].unsqueeze(0))
                    zero_encoding = torch.full((1, args.sep * (args.resize
                                                               // 32) * (args.resize // 32)), 0)
                    if torch.cuda.is_available():
                        zero_encoding = zero_encoding.cuda()

                    AB_encoding = torch.cat(
                        [common_A, zero_encoding, separate_B], dim=1)
                    AB_decoding = decoder(AB_encoding)
                    exps.append(AB_decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    if BtoA:
        vutils.save_image(exps,
                          '%s/experiments_%06d-BtoA.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display + 1)
    else:
        vutils.save_image(exps,
                          '%s/experiments_%06d-AtoB.png' % (args.out, iters),
                          normalize=True, nrow=args.num_display + 1)


def interpolate_fixed_common(args, e_common, e_separate_A, e_separate_B, decoder, imgA1, imgA2, imgB1,
                             imgB2, content_img):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    common = e_common(test_domB[content_img].unsqueeze(0))
    a1 = e_separate_A(test_domA[imgA1].unsqueeze(0))
    a2 = e_separate_A(test_domA[imgA2].unsqueeze(0))
    b1 = e_separate_B(test_domB[imgB1].unsqueeze(0))
    b2 = e_separate_B(test_domB[imgB2].unsqueeze(0))
    with torch.no_grad():
        filler = test_domB[0].unsqueeze(0).clone()
        exps.append(filler.fill_(0))
        exps.append(test_domA[imgA1].unsqueeze(0))
        for i in range(args.num_display - 2):
            exps.append(filler.fill_(0))
        exps.append(test_domA[imgA2].unsqueeze(0))

        for i in range(args.num_display):
            if i == 0:
                exps.append(test_domB[imgB1].unsqueeze(0))
            elif i == args.num_display - 1:
                exps.append(test_domB[imgB2].unsqueeze(0))
            else:
                exps.append(filler.fill_(0))

            for j in range(args.num_display):
                cur_sep_A = (float(j) / (args.num_display - 1)) * a2 + \
                            (1 - float(j) / (args.num_display - 1)) * a1
                cur_sep_B = (float(i) / (args.num_display - 1)) * b2 + \
                            (1 - float(i) / (args.num_display - 1)) * b1
                encoding = torch.cat([common, cur_sep_A, cur_sep_B], dim=1)
                decoding = decoder(encoding)
                exps.append(decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/interpolation_fixed_C.png' % (args.out),
                      normalize=True, nrow=args.num_display + 1)


def interpolate_fixed_A(args, e_common, e_separate_A, e_separate_B, decoder, imgC1, imgC2, imgB1,
                        imgB2, imgA):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    c1 = e_common(test_domB[imgC1].unsqueeze(0))
    c2 = e_common(test_domB[imgC2].unsqueeze(0))
    a = e_separate_A(test_domA[imgA].unsqueeze(0))
    b1 = e_separate_B(test_domB[imgB1].unsqueeze(0))
    b2 = e_separate_B(test_domB[imgB2].unsqueeze(0))
    with torch.no_grad():
        filler = test_domB[0].unsqueeze(0).clone()
        exps.append(filler.fill_(0))
        exps.append(test_domB[imgC1].unsqueeze(0))
        for i in range(args.num_display - 2):
            exps.append(filler.fill_(0))
        exps.append(test_domB[imgC2].unsqueeze(0))

        for i in range(args.num_display):
            if i == 0:
                exps.append(test_domB[imgB1].unsqueeze(0))
            elif i == args.num_display - 1:
                exps.append(test_domB[imgB2].unsqueeze(0))
            else:
                exps.append(filler.fill_(0))

            for j in range(args.num_display):
                cur_common = (float(j) / (args.num_display - 1)) * c2 + \
                             (1 - float(j) / (args.num_display - 1)) * c1
                cur_sep_B = (float(i) / (args.num_display - 1)) * b2 + \
                            (1 - float(i) / (args.num_display - 1)) * b1
                encoding = torch.cat([cur_common, a, cur_sep_B], dim=1)
                decoding = decoder(encoding)
                exps.append(decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/interpolation_fixed_A.png' % (args.out),
                      normalize=True, nrow=args.num_display + 1)


def interpolate_fixed_B(args, e_common, e_separate_A, e_separate_B, decoder, imgC1, imgC2, imgA1,
                        imgA2, imgB):
    test_domA, test_domB = get_test_imgs(args)
    exps = []
    c1 = e_common(test_domB[imgC1].unsqueeze(0))
    c2 = e_common(test_domB[imgC2].unsqueeze(0))
    a1 = e_separate_A(test_domA[imgA1].unsqueeze(0))
    a2 = e_separate_A(test_domA[imgA2].unsqueeze(0))
    b = e_separate_B(test_domB[imgB].unsqueeze(0))
    with torch.no_grad():
        filler = test_domB[0].unsqueeze(0).clone()
        exps.append(filler.fill_(0))
        exps.append(test_domB[imgC1].unsqueeze(0))
        for i in range(args.num_display - 2):
            exps.append(filler.fill_(0))
        exps.append(test_domB[imgC2].unsqueeze(0))

        for i in range(args.num_display):
            if i == 0:
                exps.append(test_domA[imgA1].unsqueeze(0))
            elif i == args.num_display - 1:
                exps.append(test_domA[imgA2].unsqueeze(0))
            else:
                exps.append(filler.fill_(0))

            for j in range(args.num_display):
                cur_common = (float(j) / (args.num_display - 1)) * c2 + \
                             (1 - float(j) / (args.num_display - 1)) * c1
                cur_sep_A = (float(i) / (args.num_display - 1)) * a2 + \
                            (1 - float(i) / (args.num_display - 1)) * a1
                encoding = torch.cat([cur_common, cur_sep_A, b], dim=1)
                decoding = decoder(encoding)
                exps.append(decoding)

    with torch.no_grad():
        exps = torch.cat(exps, 0)

    vutils.save_image(exps,
                      '%s/interpolation_fixed_B.png' % (args.out),
                      normalize=True, nrow=args.num_display + 1)

if __name__ == '__main__':
    pass