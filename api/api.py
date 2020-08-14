from flask import Flask, jsonify, request, make_response, render_template
import requests
import argparse
import uuid
import json
import time
from tqdm import tqdm
import threading
from queue import Queue, Empty
import base64
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.framework import ops

from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from deepface.extendedmodels import Age, Gender, Race, Emotion

#import DeepFace
#from basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
#from extendedmodels import Age, Gender, Race, Emotion

#------------------------------

app = Flask(__name__)

#------------------------------

DeepFace.allocateMemory()

tic = time.time()

print("Loading Face Recognition Models...")

pbar = tqdm(range(0,5), desc='Loading Face Recognition Models...')

for index in pbar:
	if index == 0:
		pbar.set_description("Loading VGG-Face")
		vggface_model = VGGFace.loadModel()
	elif index == 1:
		pbar.set_description("Loading OpenFace")
		openface_model = OpenFace.loadModel()
	elif index == 2:
		pbar.set_description("Loading Google FaceNet")
		facenet_model = Facenet.loadModel()
	elif index == 3:
		pbar.set_description("Loading Facebook DeepFace")
		deepface_model = FbDeepFace.loadModel()
	elif index == 4:
		pbar.set_description("Loading DeepID DeepFace")
		deepid_model = DeepID.loadModel()

toc = time.time()

print("Face recognition models are built in ", toc-tic," seconds")

#------------------------------

tic = time.time()

print("Loading Facial Attribute Analysis Models...")

pbar = tqdm(range(0,4), desc='Loading Facial Attribute Analysis Models...')

for index in pbar:
	if index == 0:
		pbar.set_description("Loading emotion analysis model")
		emotion_model = Emotion.loadModel()
	elif index == 1:
		pbar.set_description("Loading age prediction model")
		age_model = Age.loadModel()
	elif index == 2:
		pbar.set_description("Loading gender prediction model")
		gender_model = Gender.loadModel()
	elif index == 3:
		pbar.set_description("Loading race prediction model")
		race_model = Race.loadModel()

toc = time.time()

facial_attribute_models = {}
facial_attribute_models["emotion"] = emotion_model
facial_attribute_models["age"] = age_model
facial_attribute_models["gender"] = gender_model
facial_attribute_models["race"] = race_model

print("Facial attribute analysis models are built in ", toc-tic," seconds")

#------------------------------

graph = ops.get_default_graph()

#------------------------------
#Service API Interface

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

def handle_requests_by_batch():
	while True:
		requests_batch = []
		while not (len(requests_batch) >= BATCH_SIZE):
			try:
				requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
			except Empty:
				continue
			batch_outputs = []
			for request in requests_batch:
				if request['input'][1] == "analyze":
					batch_outputs.append(runAnalyze(request['input'][0]))
				else:
					batch_outputs.append(runVerify(request['input'][0]))

			for request, output in zip(requests_batch, batch_outputs):
				request['output'] = output

threading.Thread(target=handle_requests_by_batch).start()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():

	if requests_queue.qsize() >= BATCH_SIZE:
		return jsonify({"error":'Too Many Request'}), 429
	
	tic = time.time()
	# req = request.get_json()
	trx_id = uuid.uuid4()

	image = base64.b64encode(request.files['image'].read()).decode('utf-8')
	image = "data:image/jpeg;base64," + image
	print(image[:20])
	req_batch = {
		'input': [image,'analyze']
	}

	requests_queue.put(req_batch)

	while 'output' not in req_batch:
		time.sleep(CHECK_INTERVAL)
	
	resp_obj = req_batch['output']

	if 'success' in resp_obj:
		return jsonify(resp_obj), 205
	
	if 'error' in resp_obj:
		return jsonify(resp_obj), 500

	toc = time.time()
	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic

	return jsonify(resp_obj), 200

	
def runAnalyze(image):

	global graph
	
	#---------------------------

	with graph.as_default():
		instances = [image]
		# if "img" in list(req.keys()):
		# 	raw_content = req["img"] #list

		# 	for item in raw_content: #item is in type of dict
		# 		instances.append(item)
		
		if len(instances) == 0:
			return {'success': False, 'error': 'you must pass at least one img object in your request'}
		
		print("Analyzing ", len(instances)," instances")

		#---------------------------

		actions= ['emotion', 'age', 'gender', 'race']
		# if "actions" in list(req.keys()):
		# 	actions = req["actions"]
		
		#---------------------------
		try:
		#resp_obj = DeepFace.analyze(instances, actions=actions)
			resp_obj = DeepFace.analyze(instances, actions=actions, models=facial_attribute_models)
		except:
			return {'error': 'server error'}
		#---------------------------

	if 'error' in resp_obj:
		return resp_obj['error']

	return resp_obj

@app.route('/verify', methods=['POST'])
def verify():
	
	print(requests_queue.qsize())
	if requests_queue.qsize() >= BATCH_SIZE:
		return jsonify({"error":'Too Many Request'}), 429

	tic = time.time()
	# req = request.get_json()
	trx_id = uuid.uuid4()

	img1 = base64.b64encode(request.files['image1'].read()).decode('utf-8')
	img2 = base64.b64encode(request.files['image2'].read()).decode('utf-8')

	img1 = "data:image/jpeg;base64," + img1
	img2 = "data:image/jpeg;base64," + img2
	print('img1:',img1[:50])
	print('img2:',img2[:50])
	model_name = request.form['model_name']
	distance_metric = request.form['distance_metric']
	
	req = {
		'model_name': model_name,
		'distance_metric': distance_metric,
		'img': [
			{
				'img1': img1,
				'img2': img2
			}
		]
	}

	req_batch = {
		'input': [req,'verify']
	}

	requests_queue.put(req_batch)

	while 'output' not in req_batch:
		time.sleep(CHECK_INTERVAL)
	
	resp_obj = req_batch['output']

	if 'success' in resp_obj:
		return jsonify(resp_obj), 205
	
	if 'error' in resp_obj:
		return jsonify(resp_obj), 500

	toc =  time.time()
	
	resp_obj["trx_id"] = trx_id
	resp_obj["seconds"] = toc-tic
	
	return jsonify(resp_obj), 200

def runVerify(req):

	global graph
	
	with graph.as_default():
		
		model_name = "VGG-Face"; distance_metric = "cosine"

		if "model_name" in list(req.keys()):
			model_name = req["model_name"]
		if "distance_metric" in list(req.keys()):
			distance_metric = req["distance_metric"]
		
		#----------------------
		instances = []
		if "img" in list(req.keys()):
			raw_content = req["img"] #list

			for item in raw_content: #item is in type of dict
				instance = []
				img1 = item["img1"]; img2 = item["img2"]

				validate_img1 = False
				if len(img1) > 11 and img1[0:11] == "data:image/":
					validate_img1 = True
				
				validate_img2 = False
				if len(img2) > 11 and img2[0:11] == "data:image/":
					validate_img2 = True

				if validate_img1 != True or validate_img2 != True:
					return {'success': False, 'error': 'you must pass both img1 and img2 as base64 encoded string'}

				instance.append(img1); instance.append(img2)
				instances.append(instance)
		#--------------------------

		if len(instances) == 0:
			return {'success': False, 'error': 'you must pass at least one img object in your request'}
		
		#--------------------------
		try:
			if model_name == "VGG-Face":
				resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = vggface_model)
			elif model_name == "Facenet":
				resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = facenet_model)
			elif model_name == "OpenFace":
				resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = openface_model)
			elif model_name == "DeepFace":
				resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = deepface_model)
			elif model_name == "DeepID":
				resp_obj = DeepFace.verify(instances, model_name = model_name, distance_metric = distance_metric, model = deepid_model)
			elif model_name == "Ensemble":
				models =  {}
				models["VGG-Face"] = vggface_model
				models["Facenet"] = facenet_model
				models["OpenFace"] = openface_model
				models["DeepFace"] = deepface_model
				
				resp_obj = DeepFace.verify(instances, model_name = model_name, model = models)
				
			else:
				return jsonify({'success': False, 'error': 'You must pass a valid model name. Available models are VGG-Face, Facenet, OpenFace, DeepFace but you passed %s' % (model_name)})
		except:
			return {'error':'serverError'}
	#--------------------------
	
	if 'error' in resp_obj:
		return resp_obj['error']
	
	return resp_obj

@app.route('/healthz', methods=['GET'])
def checkHealth():
	return "Pong",200


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80)


