from __future__ import division
from __future__ import print_function

import requests
import os
from os import listdir
from os.path import join, isfile
from PIL import Image, ImageChops
import math
import numpy as np
import cv2
import random
import string
from scipy.misc import imread
from matplotlib import pyplot as plt

chars_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
chars_dict = {c: chars_list.index(c) for c in chars_list}

IMAGE_TOTAL = 50
RAW_PATH = "data/raw/"
SLICED_PATH = "data/sliced/"
NOISE_RM_PATH = "data/noise-removed/"
DETECTED_PATH = "data/detect/"


part = 0
path2 = 0
list_chars = [f for f in listdir('data/chars') if isfile(join('data/chars', f)) and 'jpg' in f]

def crawl_images():
	url = "https://captcha.alibaba.com/get_img?identity=aliexpress.com&sessionid=e03552bfsd3dsfbbac31f0798"
	for i in range (0, IMAGE_TOTAL):
		file_path = join(RAW_PATH,'{0:04}.jpg'.format(i))
		print(file_path)
		with open(file_path, 'wb') as f:
			response = requests.get(url)
			if response.ok: f.write(response.content)

def process_directory(directory):
    file_list = []
    for file_name in listdir(directory):
        file_path = join(directory, file_name)
        if isfile(file_path) and 'jpg' in file_name:
            file_list.append(file_path)
    return file_list

def process_image(image_path):
    image = imread(image_path)
    image = image.reshape(1080,)
    return np.array([x/255. for x in image])

def reduce_noise(file_path):
	print(file_path)
	img = cv2.imread(file_path)
	dst = cv2.fastNlMeansDenoisingColored(img,None,50,50,7,21)
	cv2.imwrite(file_path, dst)
	img = Image.open(file_path).convert('L')
	img = img.point(lambda x: 0 if x<128 else 255, '1')
	img.save(file_path)

def reduce_noise_dir(directory):
	list_file = process_directory(directory)
	for file_path in list_file:
		print(file_path)
		img = cv2.imread(file_path)
		dst = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
		cv2.imwrite(file_path, dst)
		img = Image.open(file_path).convert('L')
		img = img.point(lambda x: 0 if x<128 else 255, '1')
		img.save(file_path)

def crop(file_path, out_directory):
	part = 0
	img = Image.open(file_path)
	p = img.convert('P')
	w, h = p.size

	letters = []
	left, right= -1, -1
	found = False
	for i in range(w):
		in_letter = False
		for j in range(h):
			if p.getpixel((i,j)) == 0:
				in_letter = True
				break
		if not found and in_letter:
			found = True
			left = i
		if found and not in_letter and i-left > 25:
			found = False
			right = i
			letters.append([left, right])
	origin = file_path.split('/')[-1].split('.')[0]
	for [l,r] in letters:
		if r-l < 40:
			bbox = (l, 0, r, h)
			crop = img.crop(bbox)
			crop = crop.resize((30,60))
			crop.save(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
			part += 1

def test (file_path, out_directory, num):
	img = cv2.imread(file_path)
	global part

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

	ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
	
	kernel = np.ones((1,1), np.uint8) 
	img_dilation = cv2.dilate(thresh, kernel, iterations=1)
	
	# cv2.imshow('dilated', img_dilation) 
	# cv2.waitKey(0)
	#find contours 
	ctrs,hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	#sort contours 
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	for i, ctr in enumerate(sorted_ctrs):
		# rotate reg
		# rect = cv2.minAreaRect(ctr)
		# box = cv2.boxPoints(rect)
		# box = np.int0(box)
		# print(box)
		# cv2.drawContours(img,[box],0,(0,255,0),1)

		# contour approximation
		# epsilon = 0.1*cv2.arcLength(ctr,True)
		# approx = cv2.approxPolyDP(ctr,epsilon,True)
		# cv2.drawContours(img,[approx],0,(0,255,0),1)

    # Get bounding box 
		x, y, w, h = cv2.boundingRect(ctr)
		if w < 40:
    # Getting ROI 
    # show ROI 
		# cv2.imshow('segment no:'+str(i),roi)
		# cv2.waitKey(0)
		# cv2.rectangle(img,(x,y),( x + w, y + h ),(0,0,0),1) 
    #cv2.waitKey(0)
			roi = img[y:y+h, x:x+w] 
			# bbox = (x, y, x+w, y+h)
			# cv2.imshow('segment no:'+str(i),roi)
			# cv2.waitKey(0)
		# crop = img.crop(bbox)
			origin = file_path.split('/')[-1].split('.')[0]
			roi = cv2.resize(roi, (30, 36)) 
			cv2.imwrite(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)),roi)
			part += 1
		else:
			w2 = math.floor(w/2)
			roi = img[y:y+h, x:x + w2]
			origin = file_path.split('/')[-1].split('.')[0]
			roi = cv2.resize(roi, (30, 36)) 
			cv2.imwrite(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)),roi)
			part += 1
			# //
			roi = img[y:y+h, x+w2:x + w2*2]
			origin = file_path.split('/')[-1].split('.')[0]
			roi = cv2.resize(roi, (30, 36)) 
			cv2.imwrite(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)),roi)
			part += 1
		# cv2.imshow('marked areas',roi)
		# cv2.waitKey(0)

		# print(origin)
	# cv2.imshow('marked areas',img)
	# cv2.waitKey(0)


def test2 (file_path):
	img = cv2.imread(file_path)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

	ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
	
	# noise removal
	kernel = np.ones((1,1),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=1)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1

	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	
	cv2.imshow('marked areas',img)
	cv2.waitKey(0)


def crop_dir(raw_directory, out_directory):
	list_file = process_directory(raw_directory)
	# global part
	
	for file_path in list_file:
		# print(file_path)
		test(file_path, out_directory, part)
		# part+=1
		# img = Image.open(file_path)
		# p = img.convert('P')
		# w, h = p.size
		# letters = []
		# left, right= -1, -1
		# found = False
		# for i in range(w):
		# 	in_letter = False
		# 	for j in range(h):
		# 		if p.getpixel((i,j)) == 0:
		# 			in_letter = True
		# 			break
		# 	if not found and in_letter:
		# 		found = True
		# 		left = i
		# 	if found and not in_letter and i-left > 25:
		# 		found = False
		# 		right = i
		# 		letters.append([left, right])
		# origin = file_path.split('/')[-1].split('.')[0]
		# for [l,r] in letters:
		# 	if r-l < 40:
		# 		bbox = (l, 0, r, h)
		# 		crop = img.crop(bbox)
		# 		crop = crop.resize((60,120))
		# 		crop.save(join(out_directory, '{0:04}_{1}.jpg'.format(part, origin)))
		# 		part += 1

def adjust_dir(directory):
	list_file = process_directory(directory)
	for file_path in list_file:
		img = Image.open(file_path)
		p = img.convert('P')
		w, h = p.size
		start, end = -1, -1
		found = False
		for j in range(h):
			in_letter = False
			for i in range(w):
				if p.getpixel((i,j)) == 0:
					in_letter = True
					break
			if not found and in_letter:
				found = True
				start = j
			if found and not in_letter and j-start > 35:
				found = False
				end = j
		bbox = (0, start, w, end)
		crop = img.crop(bbox)
		crop = crop.resize((30,36))
		crop.save(file_path)

def rand_string(N=6):
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

def rename(path, filename, letter):
	os.rename(join(path,filename), join(path, letter+'-' + rand_string() + '.jpg'))

def detect_char(path, filename):
	class Fit:
		letter = None
		difference = 0
	best = Fit()
	_img = Image.open(join(path, filename))
	for img_name in list_chars:
		current = Fit()
		img = Image.open(join('data/chars', img_name))
		current.letter = img_name.split('-')[0]
		difference = ImageChops.difference(_img, img)
		for x in range(difference.size[0]):
			for y in range(difference.size[1]):
				# print(difference.getpixel((x, y)))
				current.difference += difference.getpixel((x, y))[0]/255
		if not best.letter or best.difference > current.difference:
			best = current
	if best.letter == filename.split('-')[0]: return
	print(filename, best.letter)
	rename(path, filename, best.letter)

def detect_dir(directory):
	for f in listdir(directory):
		if isfile(join(directory, f)) and 'jpg' in f:
			detect_char(directory, f)

def prepare_image(img):
    """Transform image to greyscale and blur it"""
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SMOOTH_MORE)
    if 'L' != img.mode:
        img = img.convert('L')
    return img

def remove_noise(img, pass_factor):
    for column in range(img.size[0]):
        for line in range(img.size[1]):
            value = remove_noise_by_pixel(img, column, line, pass_factor)
            img.putpixel((column, line), value)
    return img

def remove_noise_by_pixel(img, column, line, pass_factor):
    if img.getpixel((column, line)) < pass_factor:
        return (0)
    return (255)




if __name__=='__main__':
	crawl_images()
	# reduce_noise_dir(RAW_PATH)
	# crop_dir(RAW_PATH, SLICED_PATH)
	# adjust_dir(SLICED_PATH)
	# detect_dir(SLICED_PATH)
	pass
