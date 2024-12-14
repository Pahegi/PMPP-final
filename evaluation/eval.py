import os
import csv
import matplotlib.pyplot as plt

def main():
	if not os.path.exists('pdf'):
		os.makedirs('pdf')

	for file in os.listdir('csv'):
		if file.endswith('.csv'):
			name = file.split('.')[0]
			output = 'pdf/' + name + '.pdf'
			with open('csv/' + file, 'r') as f:
				reader = csv.reader(f)
				data = list(reader)
				plot_data(data, output)


def plot_data(data, output_file):
	# first row is the header and can be ignored
	# create matplotlib plot
	# first column is the x-axis (number of operators)
	# third column is the y-axis (time)
	# create one line for each second column (number of inputs)
	# save plot to output_file

	plt.plot(data[1:, 0], data[1:, 2])
	plt.savefig(output_file)
	plt.close()





if __name__ == '__main__':
	main()
