""" Plotting utils """

import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib
matplotlib.style.use("fivethirtyeight")


def plot_training_loss(training_losses, output):
	""" Plot the training loss across the iterations """
	iterations = [tl[0] for tl in training_losses]
	losses = [tl[1] for tl in training_losses]

	df = pd.DataFrame({
		"Iterations": iterations,
		"Loss": losses
	})

	ax = df.plot.line(x="Iterations", y="Loss", legend=False, style='.-' )
	ax.set_xlabel("Iterations")
	ax.set_ylabel("Loss")
	fig = ax.get_figure()
	plt.tight_layout()
	fig.savefig(output)

def plot_train_val_psnr(train_psnr, val_psnr, output):
	""" Plot the train and val psnr across the epochs """

	df = pd.DataFrame({
		"Epochs": range(len(train_psnr)),
		"Train PSNR": train_psnr,
		"Val PSNR": val_psnr
	})

	fig, ax = plt.subplots(1, 1)
	ax = df.plot.line(x="Epochs", y="Train PSNR", ax=ax)
	ax = df.plot.line(x="Epochs", y="Val PSNR", ax=ax)
	ax.set_xlabel("Epochs")
	ax.set_ylabel("Train/Val PSNR")
	plt.tight_layout()
	fig.savefig(output)
