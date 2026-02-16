import os
import time
import argparse
import matplotlib.pyplot as plt

from model import *
from data import *

#==========================================================

num_batch = 10
num_workers = 4

learning_rate = 1e-3
min_learning_rate = 1e-5
lambda1 = lambda epochs: max(0.975 ** epochs, min_learning_rate/learning_rate)
save_frequency = 10
nfilter = 64
load_first = False
alpha = 1.1
beta = 1

augment_noise = 0.025

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------

def train(args):
	# load dataset
	train_data = SignalDataset(args.data_path, 'train', noise=augment_noise, nfold=args.nfold, load_first=load_first) 
	test_data = SignalDataset(args.data_path, 'test', nfold=args.nfold, load_first=load_first)

	print('Training data:', len(train_data), 'Testing data:', len(test_data))
	
	num_batches = np.ceil(len(train_data)/num_batch).astype('int')
	num_batches += np.ceil(len(test_data)/num_batch).astype('int')

	# create model
	model = UNETDD().to(DEVICE)

	train_loader = DataLoader(train_data, batch_size=num_batch, num_workers=num_workers, shuffle=True) 
	test_loader = DataLoader(test_data, batch_size=num_batch, num_workers=num_workers)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

	mse = nn.MSELoss(reduction='mean')

	history = {'train_loss':[], 'test_loss':[], 'toc':[], 'train_dice':[], 'train_mse':[], 'test_dice':[], 'test_mse':[]}

	if args.state_file is not None:
		checkpoint = torch.load(args.state_file)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		history = checkpoint['history']
		del checkpoint

	h = len(history['train_loss'])

	for n in range(h, args.num_epoch):
		tic = time.time()

		# Training Data
		model.train()
		train_running_loss = 0.0
		train_dice = 0.0
		train_mse = 0.0
		
		for i, (signals, labels, images) in enumerate(train_loader):
			print('Epoch {0} of {1}, Batch {2} of {3} [training]...'.format(n+1,args.num_epoch,i+1,num_batches), end='\r', flush=True)

			signals = signals.to(DEVICE)
			labels = labels.to(DEVICE)
			images = images.to(DEVICE)

			optimizer.zero_grad(set_to_none=True)

			output1, output2 = model(signals)
			tdice = dice_loss(output1, labels)
			tmse = mse(output2, images)
			train_loss = ((beta)*tdice) + (alpha*tmse)
			train_loss.backward()
			optimizer.step()

			train_running_loss += train_loss.item()
			train_dice += tdice.item()
			train_mse += tmse.item()

		before_lr = optimizer.param_groups[0]["lr"]
		scheduler.step()
		after_lr = optimizer.param_groups[0]["lr"]
		print("\nEpoch {}: lr {:.2e} -> {:.2e}" .format(n+1, before_lr, after_lr))

		# Validation Data
		model.eval()
		test_running_loss = 0.0
		test_dice = 0.0
		test_mse = 0.0

		with torch.no_grad():
			for j, (signals, labels, images) in enumerate(test_loader):
				print('Epoch {0} of {1}, Batch {2} of {3} [validation]...'.format(n+1,args.num_epoch,i+j+1,num_batches), end='\r', flush=True)

				signals = signals.to(DEVICE)
				labels = labels.to(DEVICE)
				images = images.to(DEVICE)

				output1, output2 = model(signals)
				ttdice = dice_loss(output1, labels)
				ttmse = mse(output2, images)
				test_loss = ((beta)*ttdice) + (alpha*ttmse)

				test_running_loss += test_loss.item()
				test_dice += ttdice.item()
				test_mse += ttmse.item()

		model_train_loss = train_running_loss/len(train_loader)
		model_test_loss = test_running_loss/len(test_loader)
		model_train_dice = train_dice/len(train_loader)
		model_train_mse = train_mse/len(train_loader)
		model_test_dice = test_dice/len(test_loader)
		model_test_mse = test_mse/len(test_loader)

		history['train_loss'].append(model_train_loss)
		history['test_loss'].append(model_test_loss)
		history['train_dice'].append(model_train_dice)
		history['train_mse'].append(model_train_mse)
		history['test_dice'].append(model_test_dice)
		history['test_mse'].append(model_test_mse)

		toc = time.time() - tic
		history['toc'].append(toc)
		print('Epoch {0} of {1}, Train Loss: {2:.4f}, Test Loss: {3:.4f}, Time: {4:.2f} sec, Dice: {5:.4f}, MSE: {6:.4f}, DiceT: {7:.4f}, MSET:{8:.4f} '
		.format(n+1,args.num_epoch,model_train_loss,model_test_loss,toc, model_train_dice, model_train_mse, model_test_dice, model_test_mse))

		if (n % save_frequency == 0): 
			save_model(args.model_file,model,optimizer,history,'{0:02d}'.format(n))

	save_model(args.model_file,model,optimizer,history)

	if args.log_file is not None:
		with open(args.log_file, 'w') as file:
			file.write('Epoch,Train Loss,Test Loss,Time,DiceTrain,MSETrain,DiceTest,MSETest\n')
			for n in range(args.num_epoch):
				file.write('{0},{1:.4f},{2:.4f},{3:.2f},{4:.4f},{5:.4f},{6:.4f},{7:.4f}\n'.format(n+1,history['train_loss'][n],history['test_loss'][n],
				history['toc'][n],history['train_dice'][n],history['train_mse'][n],history['test_dice'][n], history['test_mse'][n]))

	print(f'Training complete - model saved to {args.model_file}')


#----------------------------------------------------------

def test(args):
	#load dataset
	test_data = SignalDataset(args.data_path,'test',nfold=args.nfold)

	num_batches = np.ceil(len(test_data)/num_batch).astype('int')

	model = UNETDD().to(DEVICE)

	# load model
	checkpoint = torch.load(args.model_file, weights_only=True)
	model.load_state_dict(checkpoint['model'])
	del checkpoint

	test_loader = DataLoader(test_data, batch_size=num_batch, num_workers=num_workers)

	mse = nn.MSELoss(reduction='mean')
	model.eval()
	test_running_loss = 0.0
	test_dice = 0.0
	test_mse = 0.0
	loss = []
	dice_array = []
	mse_array = []
	with torch.no_grad():
		for i, (signals, labels, images) in enumerate(test_loader):
			n = '\r' if i < len(test_loader)-1 else '\n'
			print('Batch {0} of {1}...'.format(i+1,num_batches), end=n, flush=True)

			signals = signals.to(DEVICE)
			labels = labels.to(DEVICE)
			images = images.to(DEVICE)

			output1, output2 = model(signals)

			ttdice = dice_loss(output1, labels)
			ttmse = mse(output2, images)
			test_loss = ((beta)*ttdice) + (alpha*ttmse)

			test_running_loss += test_loss.item()
			test_dice += ttdice.item()
			test_mse += ttmse.item()

			loss.append(test_loss)
			dice_array.append(ttdice)
			mse_array.append(ttmse)

			test_running_loss += test_loss.item()
			test_dice += ttdice.item()
			test_mse += ttmse.item()

			for j in range(output1.shape[0]):
				label_tensor = output1[j]
				label = label_tensor.permute(1,2,0).cpu().numpy() 
				label = (label*255).astype(np.uint8)

				img_tensor = output2[j]
				img = img_tensor.permute(1,2,0).cpu().numpy() 
				img = (img*255).astype(np.uint8)

				#cv2.imshow(f'image{i+1}',img)
				cv2.imwrite(f'{args.test_path}/output{i+1:04d}_label.png', label)
				cv2.imwrite(f'{args.test_path}/output{i+1:04d}_image.png', img)
				#cv2.waitKey(0)

		with open(f'{args.test_path}/loss.csv', 'w') as file:
			file.write('Order,Loss,Dice,MSE\n')
			for n in range(220):
				file.write('{0},{1:.4f},{2:.4f},{3:.4f}\n'.format(n+1,loss[n],dice_array[n],mse_array[n]))

	model_test_loss = test_running_loss/len(test_loader)
	model_dice = test_dice/len(test_loader)
	model_mse = test_mse/len(test_loader)
	print('Test Loss: {0:.4f}, Dice: {1:.4f}, MSE: {2:.5f}'.format(model_test_loss, model_dice, model_mse))

#----------------------------------------------------------

def info(args):
	import platform
	import torch
	import torchmetrics
	import torchvision
	import numpy
	import cv2

	info_list = []
	info_list.append('python {0}'.format(platform.python_version()))
	info_list.append('pytorch {0}'.format(torch.__version__))
	info_list.append('torchmetrics {0}'.format(torchmetrics.__version__))
	info_list.append('torchvision {0}'.format(torchvision.__version__))
	info_list.append('numpy {0}'.format(numpy.__version__))
	info_list.append('opencv {0}'.format(cv2.__version__))
	info_list.append('cuda {0}'.format(torch.cuda.get_device_name()))

	if args.log_file is None:
		print()
		for n in info_list:
			print(n)
		print()
	else:
		with open(args.log_file, 'w') as file:
			for n in info_list:
				file.write(n + '\n')	

#----------------------------------------------------------

if __name__ == "__main__":

	print('\n UNETDD v')

	parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32), epilog='\nFor more information, please check README.md\n', exit_on_error=False)
	parser._optionals.title = 'command arguments'

	parser.add_argument('-job', type=str, help='job name', required=True)
	parser.add_argument('-data_path', type=str, help='image dataset directory', metavar='')
	parser.add_argument('-test_path', type=str, help='test dataset directory', metavar='')
	parser.add_argument('-model_file', type=str, help='model weights file', metavar='')
	parser.add_argument('-log_file', type=str, help='log information filename', metavar='')
	parser.add_argument('-num_epoch', type=int, help='number of training epochs', metavar='')
	parser.add_argument('-state_file', type=str, help='state file', metavar='')
	parser.add_argument('-nfold', type=int, help='fold number', metavar='')

	try:
		args = parser.parse_args()
	except SystemExit:
		raise ValueError('invalid parameters')

	#----------------------------------------------------------

	if args.data_path is not None: args.data_path = os.path.abspath(args.data_path)
	if args.test_path is not None: args.test_path = os.path.abspath(args.test_path)
	if args.state_file is not None: args.state_file = os.path.abspath(args.state_file)
		
	#----------------------------------------------------------

	if args.job == 'TRAIN':
		if args.data_path is None or not os.path.exists(args.data_path):
			raise Exception('invalid data_path')
		if args.test_path is None or not os.path.exists(args.test_path):
			raise Exception('invalid test_path')
		if args.state_file is not None and not os.path.exists(args.state_file):
			raise Exception('invalid state file')
		if args.model_file is None:
			raise Exception('invalid model file')
		if args.log_file is None:
			raise Exception('invalid log file')
		if args.num_epoch is None: 
			raise Exception('invalid epoch number')
		
		train(args)

	elif args.job == 'TEST':
		if args.data_path is None or not os.path.exists(args.data_path):
			raise Exception('invalid test_path')
		if args.test_path is None or not os.path.exists(args.test_path):
			raise Exception('invalid test_path')
		if args.model_file is None or not os.path.exists(args.model_file):
			raise Exception('invalid model file')
		
		test(args)

	elif args.job == 'INFO':
		info(args)

	else:
		print('Invalid job!')
		sys.exit(-1)

	#----------------------------------------------------------

	print(args.job.capitalize() + ' job completed!')

	'''
	for j in range(output.shape[0]):
				img_tensor = output[j]
				img = img_tensor.permute(1,2,0).cpu().numpy()
				img = (img*255).astype(np.uint8)

				cv2.imshow(f'image{i+1}',img)
				cv2.waitKey(0)

				if i == 240:
				d2_output = model.d2_output.detach().cpu().numpy()  # Convert to numpy array for plotting

				plt.figure(figsize=(10, 10))
				for i in range(8): 
					plt.subplot(2, 4, i+1)  
					plt.imshow(d2_output[0, i, :, :], cmap='gray')  
					plt.axis('off')
					plt.title(f'Channel {i+1}')

				plt.tight_layout()
				plt.show()
	'''

#----------------------------------------------------------