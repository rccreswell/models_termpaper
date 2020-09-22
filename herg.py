"""Figure for misspecification in hERG model.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.join('herg'))
sys.path.append(os.path.join('herg', 'method'))
import model_ikr as m
import parametertransform
from protocols import leak_staircase as protocol_def


def main():
    # Load data
    temperature = 25.0
    file_name = 'herg25oc1'
    cell = 'A06'

    data_dir = './herg/data'

    data_file_name = file_name + '-staircaseramp-' + cell + '.csv'
    time_file_name = file_name + '-staircaseramp-times.csv'

    data = np.loadtxt(data_dir + '/' + data_file_name,
                      delimiter=',', skiprows=1)  # headers
    times = np.loadtxt(data_dir + '/' + time_file_name,
                       delimiter=',', skiprows=1)  # headers

    # Param transforms
    transform_to_model_param = \
        parametertransform.log_transform_to_model_param
    transform_from_model_param = \
        parametertransform.log_transform_from_model_param

    # Load model
    model_loc = './herg/ikr.mmt'
    model = m.Model(model_loc,
                    protocol_def=protocol_def,
                    temperature=273.15 + temperature,
                    transform=transform_to_model_param,
                    useFilterCap=False)

    # Load parameters
    param_file = \
        './herg/out/{}-iid/{}-staircaseramp-{}-solution-542811797.txt'\
        .format(file_name, file_name, cell)
    x0 = np.loadtxt(param_file)
    x0 = transform_from_model_param(x0)

    # Make figure
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(times, data, color='k', alpha=0.5, label='Data')
    ax.plot(times, model.simulate(x0, times), color='k', label='Model fit')
    ax.legend()
    ax.set_ylabel('Current')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(
        times,
        model.simulate(x0, times) - data,
        color='k',
        alpha=0.5,
        label='Residuals'
    )
    ax.legend()

    ax.set_xlabel('Times (s)')
    ax.set_ylabel('Current')

    fig.savefig('figure6.pdf')


if __name__ == '__main__':
    main()
