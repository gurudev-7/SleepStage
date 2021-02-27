import os
import numpy as np
import matplotlib.pyplot as plt



import mne
#from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

# Load the data

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

folder_path = 'sleep-edf-database-expanded-1.0.0/sleep-cassette/'
files=[]
for data_file in sorted(os.listdir(folder_path)):
    files.append(data_file)
    
    
PSG = []
HYP = []
for i in range(0,10,2):
    PSG.append(folder_path+files[i])
    HYP.append(folder_path+files[i+1])
length=len(files) 

raw_train=[]
annot_train=[]
'''
for i in range(len(PSG)-140):
    raw_train.append(mne.io.read_raw_edf(PSG[i]))
    '''
for p,h in zip(PSG,HYP):
    raw_train.append(mne.io.read_raw_edf(p))
    annot_train.append(mne.read_annotations(h))
print("******************raw_train**********************") 
print(raw_train)

'''
for i in range(len(HYP)-140):
    annot_train.append(mne.read_annotations(HYP[i]))
    '''
print("******************annotations*******************")
print(annot_train)

print(raw_train[0].info)    

'''   
folder_path1 = 'sleep-edf-database-expanded-1.0.0/sleep-telemetry/'
files1=[]
for data_file in sorted(os.listdir(folder_path1)):
    files1.append(data_file)
'''

for i in range(len(raw_train)):
    raw_train[i].set_annotations(annot_train[i], emit_warning=False)
    raw_train[i].set_channel_types(mapping)


# plot some data
for i in range(len(raw_train)):
    raw_train[i].plot(duration=60, scalings='auto')


# Extract 30s events from annotations

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}


for p,r in zip(annot_train,raw_train):
    p.crop(p[1]['onset'] - 30 * 60,p[-2]['onset'] + 30 * 60)
    r.set_annotations(p, emit_warning=False)

events_train=[]
for r in raw_train:
    events_train.append(mne.events_from_annotations(r, event_id=annotation_desc_2_event_id, chunk_duration=30.))

#events_train=list(events_train)
print("**********************Event Train***********************")
print(events_train)


# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

'''
# plot events
for i in range(len(raw_train)):
    fig = mne.viz.plot_events(events_train[i], event_id=event_id,
                          sfreq=raw_train[i].info['sfreq'],
                          first_samp=events_train[0, 0])
'''
# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


tmax=[]
for i in range(len(raw_train)):
    tmax.append( 30. - 1. / raw_train[i].info['sfreq'])  # tmax in included
    
print("***********************maxtime***********************")
print(tmax)


epochs_train=[]
for r,e,tm in zip(raw_train,events_train,tmax):
    epochs_train.append(mne.Epochs(raw=r, events=e,
                          event_id=event_id, tmin=0., tmax=tm, baseline=None))

print(epochs_train)








