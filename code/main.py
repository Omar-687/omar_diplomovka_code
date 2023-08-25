import json
import numpy as np
import matplotlib.pyplot as plt
from util import *
from controller2 import *

# Opening JSON file
DATA_FILE_CALTECH = 'acndata_sessions_acn.json'
DATA_FILE_JPL = 'acndata_sessions_jpl.json'

data,data2 = None,None
with open(DATA_FILE_CALTECH) as f:
    data = json.load(f)
with open(DATA_FILE_JPL) as f2:
    data2 = json.load(f2)






U = 10 #
W = 54 # number of chargers
X = 108 #state space R^108
time_interval = 12#time interval 12 minutes
power_rating = 150

laxitiy = 0
learning_rate = 3*10**-4
discount_factor = 0.5
relay_buffer_size = 10**6
Beta_param = 1*10**3 - 1*10**6




# a(j) denotes connecting time
# d(j) denotes disconnection time (departure time)
# e(j) - total energy delivered
# r(j) - peak charging rate
start_measurement_time = data['_meta']['start']
end_measurement_time = data['_meta']['end']
print('start ',start_measurement_time)
print('end ',end_measurement_time)
voltage = 220  # volts

# Default maximum charging rate for each EV battery.
default_battery_power = 32 * voltage / 1000  # kW

r_j = default_battery_power
arr_e = np.array([])

agg_xs_t = np.array([])
a_j,d_j,e_j = np.array([]), np.array([]), np.array([])
for i in data['_items'][:10]:
    a_j = np.append(a_j,i['connectionTime'])
    d_j = np.append(d_j,i['disconnectTime'])
    e_j = np.append(e_j,float(i['kWhDelivered']))
    c = np.array([i['connectionTime'],float(i['kWhDelivered'])])
    agg_xs_t = np.append(agg_xs_t,c)

min_timestamp = min(a_j, key=lambda x: time.mktime(time.strptime(x, "%a, %d %b %Y %H:%M:%S %Z")))


arr_e2 = np.array([])
agg2_xs_t = np.array([])
a_j2, d_j2, e_j2 = np.array([]), np.array([]), np.array([])
for i in data2['_items'][:10]:
    a_j2 = np.append(a_j2, i['connectionTime'])
    d_j2 = np.append(d_j2, i['disconnectTime'])
    e_j2 = np.append(e_j2, float(i['kWhDelivered']))
    c = [i['connectionTime'], float(i['kWhDelivered'])]
    agg2_xs_t = np.append(agg2_xs_t, c)

min_timestamp2 = min(a_j, key=lambda x: time.mktime(time.strptime(x, "%a, %d %b %Y %H:%M:%S %Z")))


input1 = np.array([a_j, d_j, e_j])
input2 = np.array([a_j2, d_j2, e_j2])


names = ['caltech', 'jpl']
values = [len(arr_e), len(arr_e2)]



dim_hid = 256
dim_in = 3
dim_out = 2

model = PPCController(dim_in, dim_hid, dim_out, 3*10**-4, min_timestamp);
# model.my_train(input1,r_j)
model.training_experimental()
# LLF(d_j,e_j,r_j)


# LLF is an optimal algorithm because if a task set will pass utilization test then it is surely schedulable by LLF. Another advantage of LLF is that it some advance knowledge about which task going to miss its deadline. On other hand it also has some disadvantages as well one is its enormous computation demand as each time instant is a scheduling event. It gives poor performance when more than one task has least laxity.



