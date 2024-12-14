import os
import csv
import matplotlib.pyplot as plt

def main():
	if not os.path.exists('pdf'):
		os.makedirs('pdf')

	for file in os.listdir('csv'):
		if file.endswith('.csv'):
			name = file.split('.')[0]
			with open('csv/' + file, 'r') as f:
				reader = csv.reader(f)
				data = list(reader)
				plot_times(data, 'pdf/' + name + '_times.pdf')
				plot_sizes(data, 'pdf/' + name + '_sizes.pdf')

def split_data_for_electrons(data):
    # Ignore the first row (header)
    header = data[0]
    data = data[1:]

    # get the unique values of the second column
    unique_inputs = sorted(set(row[1] for row in data))

    # return a list of lists, each list containing the rows with the same second column value
    return [[row for row in data if row[1] == input_value] for input_value in unique_inputs]

def plot_times(data, output_file):
    electrons = split_data_for_electrons(data)

    # plot a line for each electron
    for electron in electrons:
        x = [int(row[0]) for row in electron]
        y = [float(row[2])/1000 for row in electron]
        line, = plt.plot(x, y)
        # Add a label to the beginning of the line
        plt.annotate(f'{electron[0][1]}', xy=(x[0], y[0]), textcoords='offset points', xytext=(5,5), ha='right', color=line.get_color())

    plt.xlabel('Number of operators')
    plt.ylabel('Time (ms)')
    plt.savefig(output_file)
    plt.close()

def plot_sizes(data, output_file):
    # Ignore the first row (header)
    data = data[1:]

    # Check if each line has 4 columns
    for row in data:
        if len(row) != 4:
            print('Skipping file, invalid number of columns')
            return

    electrons = split_data_for_electrons(data)

    # plot a line for each electron
    for electron in electrons:
        x = [int(row[0]) for row in electron]
        y = [float(row[3]) for row in electron]
        line, = plt.plot(x, y)
        # Add a label to the beginning of the line
        plt.annotate(f'{electron[0][1]}', xy=(x[0], y[0]), textcoords='offset points', xytext=(5, 5), ha='right',
                     color=line.get_color())

    plt.xlabel('Number of operators')
    plt.ylabel('waveform size')
    plt.savefig(output_file)
    plt.close()



if __name__ == '__main__':
	main()
