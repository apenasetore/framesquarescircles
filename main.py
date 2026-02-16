import os
import time
import argparse
import matplotlib.pyplot as plt

from model import *
from model import *
from data import *
from torch.utils.data import DataLoader



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

	train_data = FramePictures(args.data_path, noise=augment_noise, nfold=args.nfold, load_first=load_first) 
	test_data = FramePictures(args.test_path, noise=augment_noise, nfold=args.nfold, load_first=load_first)

	print('Training data:', len(train_data), 'Testing data:', len(test_data))
	
	num_batches = np.ceil(len(train_data)/num_batch).astype('int')
	num_batches += np.ceil(len(test_data)/num_batch).astype('int')

	# create model
	model = SVSUNET(in_c=1,nfilter=8).to(DEVICE)

	train_loader = DataLoader(train_data, batch_size=num_batch, num_workers=num_workers, shuffle=True) 
	test_loader = DataLoader(test_data, batch_size=num_batch, num_workers=num_workers)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


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
		
		for i, (images, labels) in enumerate(train_loader):
			print('Epoch {0} of {1}, Batch {2} of {3} [training]...'.format(n+1,args.num_epoch,i+1,num_batches), end='\r', flush=True)

			labels = labels.to(DEVICE)
			images = images.to(DEVICE)

			optimizer.zero_grad(set_to_none=True)
			
			output1 = model(images)

			tdice = dice_loss(output1, labels)

			train_loss = ((beta)*tdice) 
			train_loss.backward()
			optimizer.step()

			train_running_loss += train_loss.item()
			train_dice += tdice.item()

		before_lr = optimizer.param_groups[0]["lr"]
		scheduler.step()
		after_lr = optimizer.param_groups[0]["lr"]
		print("\nEpoch {}: lr {:.2e} -> {:.2e}" .format(n+1, before_lr, after_lr))

		model.eval()
		test_running_loss = 0.0
		test_dice = 0.0
		test_mse = 0.0

		with torch.no_grad():
			for j, (images, labels) in enumerate(test_loader):
				print('Epoch {0} of {1}, Batch {2} of {3} [validation]...'.format(n+1,args.num_epoch,j+1,len(test_loader)), end='\r', flush=True)

				labels = labels.to(DEVICE)
				images = images.to(DEVICE)

				output1 = model(images)

				tdice = dice_loss(output1, labels)

				test_loss = ((beta)*tdice)
				test_running_loss += test_loss.item()
				test_dice += tdice.item()


		model_train_loss = train_running_loss/len(train_loader)
		model_test_loss = test_running_loss/len(test_loader)
		model_train_dice = train_dice/len(train_loader)
		model_test_dice = test_dice/len(test_loader)
		model_test_mse = test_mse/len(test_loader)

		history['train_loss'].append(model_train_loss)
		history['test_loss'].append(model_test_loss)
		history['train_dice'].append(model_train_dice)
		history['test_dice'].append(model_test_dice)
		history['test_mse'].append(model_test_mse)

		toc = time.time() - tic
		history['toc'].append(toc)
		print('Epoch {0} of {1}, Train Loss: {2:.4f}, Test Loss: {3:.4f}, Time: {4:.2f} sec, Dice: {5:.4f}, Test Dice: {6:.4f}'
		.format(n+1,args.num_epoch,model_train_loss,model_test_loss,toc, model_train_dice, model_test_dice))

		if (n % save_frequency == 0): 
			save_model(args.model_file,model,optimizer,history,'{0:02d}'.format(n))

	save_model(args.model_file,model,optimizer,history)

	if args.log_file is not None:		
		with open(args.log_file, 'w') as file:
			file.write('Epoch,Train Loss,Test Loss,Time,DiceTrain,MSETrain,DiceTest,MSETest\n')
			
			for n in range(args.num_epoch):
				file.write(
					f"{n+1},"
					f"{history['train_loss'][n]:.4f},"
					f"{history['test_loss'][n]:.4f},"
					f"{history['toc'][n]:.2f},"
					f"{history['train_dice'][n]:.4f},"
					f"{history['train_mse'][n] if len(history['train_mse']) > n else 0.0:.4f},"
					f"{history['test_dice'][n]:.4f},"
					f"{history['test_mse'][n]:.4f}\n"
				)

	print(f'Training complete - model saved to {args.model_file}')


#----------------------------------------------------------

def test(args):
	#load dataset
	test_data = FramePictures(args.test_path, noise=augment_noise, nfold=args.nfold, load_first=load_first)

	num_batches = np.ceil(len(test_data)/num_batch).astype('int')
	model = SQUNET(in_c=1,nfilter=8).to(DEVICE)

	# load model
	checkpoint = torch.load(args.model_file, weights_only=True)
	model.load_state_dict(checkpoint['model'])
	del checkpoint

	test_loader = DataLoader(test_data, batch_size=num_batch, num_workers=num_workers)
	print('Testing data:', len(test_data))

	model.eval()
	test_running_loss = 0.0
	test_dice = 0.0
	test_mse = 0.0
	loss = []
	dice_array = []
	with torch.no_grad():
		for i, (images, labels) in enumerate(test_loader):

			n = '\r' if i < len(test_loader)-1 else '\n'
			print(f'Batch {i+1} of {num_batches}...', end=n, flush=True)

			labels = labels.to(DEVICE)
			images = images.to(DEVICE)

			output1 = model(images)

			ttdice = dice_loss(output1, labels)
			test_loss = beta * ttdice

			test_running_loss += test_loss.item()
			test_dice += ttdice.item()

			loss.append(test_loss)
			dice_array.append(ttdice)

			B = output1.shape[0]
			T = images.shape[2]  

			for j in range(B):

				# ---- Collect frames ----
				frame_list = [] 
				for k in range(T):

					frame = images[j,:,k]                 # (C,H,W)
					frame = frame.permute(1, 2, 0).cpu().numpy()
					frame = (frame * 255).astype(np.uint8)
					frame_list.append(frame)
				
				# ---- Prediction ----
				pred = output1[j]                        # (1,H,W)
				img_pred = pred.permute(1, 2, 0).cpu().numpy()
				img_pred = (img_pred * 255).astype(np.uint8)


				# ---- Label ----
				label = labels[j]
				img_label = label.permute(1, 2, 0).cpu().numpy()
				img_label = (img_label * 255).astype(np.uint8)


				# ---- Final concatenation ----
				frames_concat = np.concatenate(frame_list, axis=1)

				final_image = np.concatenate(
					[frames_concat, img_label, img_pred],
					axis=1
				)

				new_img = format_final_image(final_image)
				
				# Save
				cv2.imwrite(f"{args.results_path}/batch{i+1:04d}_sample{j:02d}_legend.png", new_img)

			with open(f'{args.results_path}/loss.csv', 'w') as file:
				file.write('Order,Loss,Dice,MSE\n')
				for n in range(len(loss)):
					file.write('{0},{1:.4f},{2:.4f}\n'.format(n+1,loss[n].item(),dice_array[n].item()))

		model_test_loss = test_running_loss/len(test_loader)
		model_dice = test_dice/len(test_loader)
		model_mse = test_mse/len(test_loader)
		print('Test Loss: {0:.4f}, Dice: {1:.4f}, MSE: {2:.5f}'.format(model_test_loss, model_dice, model_mse))


#----------------------------------------------------------
def format_final_image(img): 
	
	#This is chatgpteed to the maximum. Whatever gril, it slaay
	# Parameters
	num_cols = 6  # 4 frames + 1 label/pred
	legend_height = 30  # space at the bottom for text
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.5
	color = (255, 255, 255)  # white
	thickness = 1

	# Add extra space at the bottom for legend
	h, w, c = img.shape
	new_img = np.zeros((h + legend_height, w, c), dtype=np.uint8)
	new_img[:h, :, :] = img

	# Texts for each column
	texts = ["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Label", "Pred"]

	# Compute width of each column
	col_width = w // num_cols

	# Put text centered under each column
	for i, text in enumerate(texts):
		(text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
		x = i * col_width + (col_width - text_w) // 2
		y = h + legend_height - 5  # a few pixels from bottom
		cv2.putText(new_img, text, (x, y), font, font_scale, color, thickness)

	return new_img
#----------------------------------------------------------
if __name__ == "__main__":

	print('\n SQUNET')

	parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32), epilog='\nFor more information, please check README.md\n', exit_on_error=False)
	parser._optionals.title = 'command arguments'

	parser.add_argument('-job', type=str, help='job name', required=True)
	parser.add_argument('-data_path', type=str, help='image dataset directory', metavar='')
	parser.add_argument('-test_path', type=str, help='test dataset directory', metavar='')
	parser.add_argument('-results_path', type=str, help='results directory', metavar='')
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
		if args.test_path is None or not os.path.exists(args.test_path):
			raise Exception('invalid test_path')
		if args.model_file is None or not os.path.exists(args.model_file):
			raise Exception('invalid model file')
		
		test(args)
	else:
		print('Invalid job!')
		sys.exit(-1)

	#----------------------------------------------------------

	print(args.job.capitalize() + ' job completed!')

	

#----------------------------------------------------------