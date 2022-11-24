import argparse
import os
import json
import math

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

def look_at(eye, at, up):
	print("Lookat")
	# Direction
	zaxis = np.subtract(at, eye)
	zaxis = zaxis / np.linalg.norm(zaxis)

	# Left
	xaxis = np.cross(up, zaxis)
	xaxis = xaxis / np.linalg.norm(xaxis)

	# New up
	yaxis = np.cross(zaxis, xaxis)

	print("Look at:")
	print(xaxis)
	print(yaxis)
	print(zaxis)
	translation = np.array([
		[xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
		[yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
		[zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, eye)],
		[0.0, 0.0, 0.0, 1.0]
	])
	print("Transforms")
	print(translation)

	return translation

def look_at_inv(eye, at, up):
	print("Lookat(inverse)")
	# Direction
	zaxis = np.subtract(at, eye)
	zaxis = zaxis / np.linalg.norm(zaxis)

	# Left
	xaxis = np.cross(up, zaxis)
	xaxis = xaxis / np.linalg.norm(xaxis)

	# New up
	yaxis = np.cross(zaxis, xaxis)

	print("Look at:")
	print(xaxis)
	print(yaxis)
	print(zaxis)
	linear_transformation = np.array([
		[xaxis[0], yaxis[0], zaxis[0], 0],
		[xaxis[1], yaxis[1], zaxis[1], 0],
		[xaxis[2], yaxis[2], zaxis[2], 0],
		[0.0, 0.0, 0.0, 1.0]
	])
	translation = np.array([
		[1, 0, 0, eye[0]],
		[0, 1, 0, eye[1]],
		[0, 0, 1, eye[2]],
		[0, 0, 0, 1]
	])
	print("Transforms")
	print(linear_transformation)
	print(translation)

	return np.matmul(translation, linear_transformation)

def find_perspective_node(root):
	for sensor in root.iter('sensor'):
		if sensor.get('type') != 'perspective':
			continue
		return sensor

def find_toworld_transform_node(perspective):
	for transform in perspective.iter('transform'):
		if transform.get('name') != "toWorld":
			continue
		else:
			return transform
	return None

def find_mitsuba_attribute(node, name):
	named_nodes = node.findall(".//*[@name='%s']" % name)

	if len(named_nodes) != 1:
		return None

	return named_nodes[0]

def matrix_4x4_from_3x3(mat):
	return np.array([
		[mat[0][0], mat[0][1], mat[0][2], 0],
		[mat[1][0], mat[1][1], mat[1][2], 0],
		[mat[2][0], mat[2][1], mat[2][2], 0],
		[0, 0, 0, 1]
	])

def swap_vector_axis(vec):
	return np.array([vec[2], vec[0], vec[1]])

def build_view_matrix(transform):
	accumulated_transform = np.identity(4)
	for entry in transform:
		if entry.tag == "lookat":
			print("lookat")

			target = np.fromstring(entry.get('target'), dtype=float, sep=', ')
			origin = np.fromstring(entry.get('origin'), dtype=float, sep=', ')
			up = np.fromstring(entry.get('up'), dtype=float, sep=', ')

			print(origin)
			print(target)
			print(up)
			lookat_transform = look_at_inv(origin, target, up)

			print("Derived lookat:")
			print(lookat_transform)

			accumulated_transform = np.matmul(lookat_transform, accumulated_transform)
			#accumulated_transform = lookat_transform
		elif entry.tag == "rotate":
			print("rotate")
			if 'z' in entry.attrib and entry.get('z') == "1":
				rotation = Rotation.from_euler('z', float(entry.get('angle')), degrees=True).as_matrix()
			elif 'y' in entry.attrib and entry.get('y') == "1":
				rotation = Rotation.from_euler('y', float(entry.get('angle')), degrees=True).as_matrix()
			elif 'x' in entry.attrib and entry.get('x') == "1":
				rotation = Rotation.from_euler('x', float(entry.get('angle')), degrees=True).as_matrix()
			else:
				raise RuntimeError("rotate: Not sure what axis to rotate")
			print(rotation)
			rotation_4 = matrix_4x4_from_3x3(rotation)
			print(rotation_4)
			accumulated_transform = np.matmul(np.linalg.inv(rotation_4), accumulated_transform)
		else:
			raise RuntimeError("Unknown transform tag %s" % entry.tag)
	return accumulated_transform

def get_relative_path(image_file, transforms):
	transforms_location = os.path.dirname(transforms)
	return "./%s" % os.path.relpath(image_file, transforms_location)

def main():
	parser = argparse.ArgumentParser(description='Converts a mitsuba 0.6 scene file to camera transforms that can be used with NERF')
	parser.add_argument('scene', type=str, help='Duration to run the model')
	parser.add_argument('imagefile', type=str, help='Output image file to describe')
	parser.add_argument('transforms', type=str, help='Output transforms.json file')

	args = parser.parse_args()

	scene = ET.parse(args.scene).getroot()

	perspective_node = find_perspective_node(scene)

	# Initialize common parameters if transform.json doesn't exist
	if not os.path.exists(args.transforms):
		film = perspective_node.find('film')
		w = float(find_mitsuba_attribute(film, 'width').get('value'))
		h = float(find_mitsuba_attribute(film, 'width').get('value'))
		fov = float(find_mitsuba_attribute(perspective_node, 'fov').get('value'))
		fov_r = fov * math.pi / 180

		fl_node = find_mitsuba_attribute(perspective_node, 'focusLength')

		fl = float(fl_node.get('value')) if fl_node is not None else 50.0

		transforms = {
			"camera_angle_x": fov_r,
			"camera_angle_y": fov_r,
			"w": w,
			"h": h,
			#"fl_x": fl,
			#"fl_y": fl,
			#"k1": 0,
			#"k2": 0,
			#"p1": 0,
			#"p2": 0,
			"cx": w/2,
			"cy": h/2,
			"aabb_scale": 8,
			"frames": []
		}
	else:
		with open(args.transforms, 'r') as f:
			transforms = json.load(f)
	
	# Calculate relative path
	relative_image_path = get_relative_path(args.imagefile, args.transforms)
	print("Generating transform matrix for %s" % relative_image_path)
	for entry in transforms['frames']:
		if entry['file_path'] == relative_image_path:
			print("Data for this frame already exists, deleting")
			transforms['frames'].remove(entry)

	transform_list = find_toworld_transform_node(perspective_node)
	transform = build_view_matrix(transform_list)
	print("Final transform matrix:")
	"""
	transform = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])
	"""
	#transform = axis_transform
	#transform = np.matmul(np.matmul(translation_matrix, axis_transform_x), axis_transform_y)

	transform[0:3,2] *= -1 # flip the y and z axis
	transform[0:3,1] *= -1
	transform=transform[[1,0,2,3],:] # swap y and z
	transform[2,:] *= -1 # flip whole world upside down

	print(transform)

	#print("View-projection matrix:")
	#vp = np.matmul(transform, projection_transform)
	#print(vp)

	transforms['frames'].append({
		"file_path": relative_image_path,
		"sharpness": 2, # TODO: WTF is this
		"transform_matrix": transform.tolist()
	})

	with open(args.transforms, 'w') as f:
		json.dump(transforms, f, indent=4)


if __name__ == "__main__":
	main()
