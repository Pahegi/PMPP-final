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
    # Ignore the first row (header)
    header = data[0]
    data = data[1:]

    # Extract unique values from the second column (number of inputs)
    unique_inputs = sorted(set(row[1] for row in data))

    # Create a plot for each unique value in the second column
    for input_value in unique_inputs:
        x = [int(row[0]) for row in data if row[1] == input_value]
        y = [float(row[2])/1000 for row in data if row[1] == input_value]
        line, = plt.plot(x, y)
        # Add a label to the beginning of the line
        plt.annotate(f'{input_value}', xy=(x[0], y[0]), textcoords='offset points', xytext=(5,5), ha='right', color=line.get_color())

    plt.xlabel('Number of operators')
    plt.ylabel('Time (ms)')
    plt.savefig(output_file)
    plt.close()





if __name__ == '__main__':
	main()
