"""
	Transforms to be applied to numpy arrays with format of (H W C)
"""
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.ndimage.filters import gaussian_filter, convolve, median_filter
from skimage import transform
from skimage import exposure
import numpy as np
import torch
import data_utils.transform_2d as tf
from data_utils.transform_2d import PadToSize, Rescale
from data_utils.bbox import im_border
from random import choice, randint, random, choices, shuffle
import cv2
import math
from english_no_swear_10k import WORDS_10K

import os
import os.path as osp
import pathlib

pad = PadToSize(100, padding_intensity=1.0)
SPACE_RANGE = (20, 60)
rescale = Rescale((60, 880), False)


class GenerateTextWBox:
    def __init__(self, fonts_path, locations, line_gap, font_size):
        # ignore_fonts = ['PTS56F.ttf', 'Aller_BdIt.ttf', 'Aller_LtIt.ttf', 'Aller_It.ttf', 'Mermaid Swash Caps.ttf']
        # fpath = pathlib.Path(fonts_path)
        # all_fonts = fpath.glob('**/*.ttf')
        # self.all_fonts = []
        # for font in all_fonts:
        #     fontname = osp.basename(font)
        #     if ('Italic' not in fontname) and (fontname not in ignore_fonts):
        #         self.all_fonts.append(font)

        self.all_fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]

        self.font_size = font_size
        self.line_gap = line_gap
        
        self.current_font = choice(self.all_fonts)
        self.locations = locations

    def get_text_bbox(self, texts, bbox, locations=None):
        ymin, xmin, ymax, xmax = bbox
        h, w = (ymax - ymin), (xmax - xmin)

        im = np.ones((250, 600), dtype='uint8')*255
        lb = np.zeros_like(im)

        lines = texts.split('\n')

        position = [50, 50]
        color = 0
        thickness = 1

        if len(lines) == 1:
            im = cv2.putText(im, lines[0], tuple(position), self.current_font, self.font_size, color, thickness, cv2.LINE_AA)
            ymin, xmin, ymax, xmax = im_border(im, black_obj=True)
            lb[ymin:ymax, xmin:xmax] = 1
        else:
            for idx, line in enumerate(lines):
                tmpim = cv2.putText(np.ones_like(im)*255, line, (50, 50), self.current_font, self.font_size, color, thickness, cv2.LINE_AA)
                ymin, xmin, ymax, xmax = im_border(tmpim, black_obj=True)
                hline, wline = ymax-ymin, xmax-xmin
                
                locy, locx = position
                im[locy:locy+hline, locx:locx+wline] = tmpim[ymin:ymax, xmin:xmax] # np.min([tmpim[ymin:ymax, xmin:xmax], im[locy:locy+hline, locx:locx+wline]], axis=0)
                lb[locy:locy+hline, locx:locx+wline] = idx+1

                position[0] += (hline + self.line_gap) ## change to new
        
        ymin, xmin, ymax, xmax = im_border(im, black_obj=True)
        im = im[ymin:ymax, xmin:xmax]
        lb = lb[ymin:ymax, xmin:xmax]

        im = crop_im_to_maxsize(im, h, w)
        lb = crop_im_to_maxsize(lb, h, w)
        th, tw = im.shape

        if locations is None:
            pad_y = randint(0, h-th)
            pad_x = randint(0, w-tw)
        else:
            gap_x, gap_y = (w-tw), (h-th)
            shiftx = gap_x//8
            shifty = gap_y//15
            if locations['x'] == 'center':
                pad_x = gap_x//2
            elif locations['x'] == 'left':
                pad_x = shiftx
            elif locations['x'] == 'right':
                pad_x = gap_x - shiftx
            
            if locations['y'] == 'center':
                pad_y = (h-th)//2
            elif locations['y'] == 'top':
                pad_y = shifty
            elif locations['y'] == 'bottom':
                pad_y = gap_y - shifty
            
            pad_y += randint(-shifty, shifty)
            pad_x += randint(-shiftx, shiftx)
        # print('padding', (pad_y, h-th-pad_y), (pad_x, w-tw-pad_x))

        im = np.pad(im, [(pad_y, h-th-pad_y), (pad_x, w-tw-pad_x)], 'constant', constant_values=255)
        lb = np.pad(lb, [(pad_y, h-th-pad_y), (pad_x, w-tw-pad_x)], 'constant', constant_values=0)


        return im, lb


def crop_im_to_maxsize(im, h, w):
    th, tw = im.shape[:2]
    if th >= h:
        im = im[:int(h//1.05), :]
    if tw >= w:
        im = im[:, :int(w//1.1)]

    return im



def generate_synthetic(ims_n_space):
    maxh = 4
    for im in ims_n_space:
        if hasattr(im, 'shape'):
            maxh = max(maxh, im.shape[0])
    
    padded =[]
    for im in ims_n_space:
        if hasattr(im, 'shape'):
            w = im.shape[1]
            padded.append(pad(im, (maxh, w)))
        else:
            w = np.random.randint(*SPACE_RANGE)
            padded.append(np.full((maxh, w, 1), 1.0))
    
    return np.concatenate(padded, axis=1)

def combine_ims(ims_n_space, ref=''): ## use center of mass
    maxh = 4
    letter_space= randint(-4, 4)
    for im in ims_n_space:
        if hasattr(im, 'shape'):
            maxh = max(maxh, im.shape[0])
    
    out_im = np.full((maxh, 3200), 2.0, dtype='float32')

    current = 20
    out_im[:, :current+5] = 1.0

    for im in (ims_n_space):
        if hasattr(im, 'shape'):
            w = im.shape[1]
            tmp = pad(im, (maxh, w))
            space = letter_space+randint(-2, 2)
            if space > 0: out_im[:, current:current+space] = 1.0
            current += space
            tmp2 = out_im[:, current:current+w]
            out_im[:, current:current+w] = np.minimum(tmp, tmp2)
        else:
            w = np.random.randint(*SPACE_RANGE)
            out_im[:, current:current+w] = 1.0
        current += int(w)
    out_im[:, current:current+25] = 1.0
    out_im = out_im[:, :current+20]

    return out_im
    

def shrink_im(im):
    mx = im.max()
    line = im.min(axis=1) < 0.85*mx
    line = np.where(line)
    if isinstance(line, tuple): line = line[0]
    ymin, ymax = max(line[0], 0), line[-1]+1

    line = im.min(axis=0) < 0.85*mx
    line = np.where(line)
    if isinstance(line, tuple): line = line[0]
    xmin, xmax = max(line[0], 0), line[-1]+1
    im = im[ymin : ymax , xmin: xmax]
    return im


def text_to_image(text): ## Add multi lines
    font = cv2.FONT_HERSHEY_TRIPLEX

    im = np.ones((100, 1600))
    color = [0.]*3
    thickness = 1

    im = cv2.putText(im, text, (50, 60), font, 1.0, color, thickness, cv2.LINE_AA)
    im = rescale(im)

    line = im.min(axis=1) < 0.7
    line = np.where(line)
    if isinstance(line, tuple): line = line[0]
    ymin, ymax = max(line[0]-3, 0), line[-1]+3
    
    line = im.min(axis=0) < 0.7
    line = np.where(line)
    if isinstance(line, tuple): line = line[0]
    xmin, xmax = max(line[0]-3, 0), line[-1]+3
    im = im[ymin : ymax , xmin: xmax]

    return im

def multiline_to_image(text, line=2):
    length = math.ceil(len(text) / line)

    imgs = []
    width = 0

    i = 0

    while i < len(text):
        j = int(i+length)
        for _ in range(18):
            if j >=len(text) or text[j] == ' ': break
            else: j += 1
        subtext = text[i:j]
        im = text_to_image(subtext)
        width = max(im.shape[1], width)
        imgs.append(im)
        i=j
    
    imgs = [pad(im, target_size=(im.shape[0]+4, width)) for im in imgs]

    return np.concatenate(imgs, axis=0)

ALPHAS = 'abcdefghijklmnopqrstuvwxyz'
NUMS = '0123456789'

def number(len_range=(1, 5), suffix=None, add_symbols=''):
    length = randint(*len_range)
    ns = choices(NUMS, k=length)

    if len(add_symbols) > 0 and random() > 0.5:
        sym = choice(add_symbols)
        nums = []
        for i, n in enumerate(ns):
            if i < len(ns)-1 and random() > 0.7:
                nums += [n, sym]
            else: nums.append(n)
    else: nums = ns
    
    ns = "".join(nums)
    
    if suffix is not None:
        if isinstance(suffix, (list, tuple)):
            suffix = choice(suffix) if len(suffix) > 0 else ""
        ns += suffix
    return ns

def phone():
    separator = [' ', '.', '-', '']
    separator = choice(separator)

    rand = random()
    if rand < 0.3:
        arr = ['+'+number((1,3))]
        if separator == '': arr[0]+=' '
    elif rand < 0.3:
        arr = [number((1,3))]
        if separator == '': arr[0]+=' '
    else:
        arr=[]

    arr += [number((3,3)), number((3,3)), number((4,4))]
    
    return separator.join(arr)

def decimal(main_len=3):
    has_dec = random() > 0.7
    dec = str(randint(0, 99))
    main = str(randint(0, 10**main_len - 1))
    dec = '0'+dec if len(dec) == 1 and random()>0.5 else dec
    if random() <0.05: dec = '00'

    return '{}.{}'.format(main, dec) if has_dec else main

def fee():
    fee = decimal(main_len=3) if random() > 0.5 else '$'+decimal(main_len=3)

    ## Add optional 0s for decimals
    return fee

def teeth(num=1):
    separator = choice([',', ', ', ' '])

    babytooth = 'abcdefghijklmnopqrst'
    if random() > 0.7: babytooth = babytooth.upper()

    arr = []
    for _ in range(num):
        tooth = str(randint(1, 32)) if random()>0.7 else choice(babytooth)
        arr.append(tooth)
    return separator.join(arr)

def date():
    separator = choice(['-', '/', ',', '.', ' '])
    year = str(randint(1900, 2046))
    month = str(randint(1, 12))
    day = str(randint(1, 31))

    if random() > 0.5:
        month = month if len(month) > 1 else '0'+month
        day = day if len(day) > 1 else '0'+day
    
    formating = choice(['day', 'month', 'year'])

    if formating == 'month':
        return '{1}{0}{2}{0}{3}'.format(separator, month, day, year)
    elif formating == 'year':
        return '{1}{0}{2}{0}{3}'.format(separator, year, month, day)
    else:
        return '{1}{0}{2}{0}{3}'.format(separator, day, month, year)


def checkbox():
    if random() < 0.6:
        return ''
    else:
        return choice(['v', 'x', 'o'])

def address(parts='all'):
    '''
    could also be 'street', 'city', 'state', 'zip'
    '''

    if parts == 'all': parts = ['street', 'city', 'state', 'zip']
    elif isinstance(parts, str): parts = [parts]

    results = []

    for part in parts:
        if part == 'street':
            streets = ['st.', 'st', 'street', 'ave', 'ave.', 'avenue', 'blvd.', 'blvd', 'boulevard', word(len_range=(3, 8))]
            street = ' '.join([number(), words((1, 3)), choice(streets)])
            results.append(street)
            if random() > 0.5:
                apt = choice(['#', 'apt', 'apt.', 'APT', 'unit', 'num'])
                if random() > 0.5: apt += ' '
                apt += number([1, 3])+word(len_range=[0,1])
                results.append(apt)
        elif part == 'city':
            results.append(words((1, 3)))
        elif part=='state':
            rand = random()
            if rand < 0.3:
                state = word(len_range=[2,2]).upper()
            elif rand < 0.5:
                state = word(len_range=[2,2])
            elif rand < 0.7:
                state = word(len_range=[4, 10])
            else:
                state = word(len_range=[4, 10]).capitalize()
            results.append(state)
        elif part == 'zip':
            zipcode = number(len_range=[5, 5])
            # if random() > 0.5: zipcode = '-'+zipcode
            results.append(zipcode)
    
    address = choice([', ', '; ', ' ']).join(results)
    
    rand = random()
    if rand < 0.5:
        address = address.title()
    elif rand < 0.7:
        address = address.upper()

    return address

def name(len_range=(2, 4)):
    first = word(len_range=[2,12], word_type='alpha+symbol', add_symbols='-')
    mid = "" if random() > 0.5 else word(len_range=[1, 1], suffix='.')
    last = word(len_range=[2,12], word_type='alpha+symbol', add_symbols='-')

    title = "" if random() > 0.5 else word(len_range=[2,6], suffix='.')

    name = " ".join([first, mid, last, title]).split()
    name = ' '.join(name)

    if random() > 0.5:
        return name.upper()
    else:
        return name.title()

def word(cap='', word_type='alpha', len_range=[1, 15], add_symbols='-.', suffix=''):
    '''
    cap could also be 'first', 'all'
    type could also be 'mix', 'alpha+num', 'num-', '-num', 'alpha+symbol'
    '''
    dic = ALPHAS
    word = ''
    use_sym = False

    if word_type == 'num-':
        word = number((0,5))
    if word_type == 'alpha+num':
        dic += NUMS
    elif word_type == 'mix':
        dic += NUMS
        use_sym = True
    elif word_type == 'alpha+symbol':
        # dic += add_symbols
        use_sym = True

    letters_ = choices(dic, k=randint(*len_range))
    if use_sym and len(add_symbols) > 0 and random() > 0.5:
        sym = choice(add_symbols)
        letters = []
        for i, n in enumerate(letters_):
            if i < len(letters_)-1 and random() > 0.8:
                letters += [n, sym]
            else: letters.append(n)
    else: letters = letters_
    
    word += ''.join(letters)
    if len(word) > 0:
        if word[0] in add_symbols:
            word = choice(ALPHAS) + word[1:]
        if word[-1] in add_symbols:
            word = word[:-1] + choice(ALPHAS)
    


    if suffix:
        if isinstance(suffix, str): suffix = [suffix]
        suffix = list(suffix)
        suffix.append('')
        word += choice(suffix)
    
    if len(cap) > 0:
        if cap == 'all': cap = ['first', 'upper', '']
        if isinstance(cap, str): cap = [cap]
        c = choice(cap)
        if c == 'first':
            word = word.capitalize()
        elif c == 'upper':
            word = word.upper()
    
    return word

def words(len_range=(2, 6), word_len=(2, 10), mixed=False, 
        mix_symbols='-+/&~', cap='', use_iam=False
        ): ## 10~20 for long
    '''
    return arr of words
    '''
    words = []
    length = randint(*len_range)

    for _ in range(length):
        if mixed is False:
            rand = random()
            if rand < 0.6:
                words.append(choice(WORDS_10K))
            else:
                words.append(word(len_range=word_len))
        else:
            rand = random()
            if rand<0.5:
                w = '!?!' if use_iam else choice(WORDS_10K)
            elif rand < 0.8:
                w = word(len_range=word_len)
            elif rand < 0.9:
                w = word(word_type='mix', len_range=word_len, add_symbols=mix_symbols)
            else:
                w = number(len_range=word_len)

            if random()<0.15 and length > 1:
                w += choice(',.;:')
            words.append(w)

    words = " ".join(words)

    if len(cap) > 0:
        if cap == 'all': cap = ['first', 'upper', 'title', '']
        if isinstance(cap, str): cap = [cap]
        c = choice(cap)
        if c == 'first':
            words = words.capitalize()
        elif c == 'upper':
            words = words.upper()
        elif c == 'title':
            words = words.title()

    return words

def email():
    email = '{}@{}.{}'.format(word(word_type='mix', add_symbols='-_'), word(word_type='mix', add_symbols='-'), word(len_range=(2, 6)))

    rand = random()
    return email if rand > 0.7 else email.upper()



def generate_random_text():
    categories = 6
    text_type = randint(0, categories)
    
    if text_type == 0: # • Words
        text = words(cap='all', len_range=(1, 5), word_len=(1, 8), mixed=True, use_iam=True)
    elif text_type == 1: # • Word with all types
        text = words(cap='all', len_range=(1, 1), word_len=(1, 6), mixed=True, mix_symbols='-:.')
    elif text_type == 2: # • For letter+num mix
        text = word(cap=['', 'upper'], word_type='alpha+num', len_range=[6, 10])
    elif text_type == 3: # • Numbers (dates, phone numbers) 0
        rand = random()
        if rand < 0.6:
            text = number((1, 12), suffix=['', '', '', '', 'mo', 'months'], add_symbols='- ')
        elif rand < 0.8:
            text = phone()
        else:
            text = date()
    elif text_type == 4: # • Decimal (area, fee) 1
        text = decimal(main_len=4)
        rand = random()
        if rand > 0.8:
            text += 'mm'
        elif rand > 0.6:
            text = '$'+text
	# • Address ??
    elif text_type == 5: # tooth 5
        text = teeth(randint(1, 6))
    elif text_type == 6: # Email 6
        text = email()
    out_type = np.zeros(categories+1, dtype='bool')
    out_type[text_type] = True

    return text, out_type
