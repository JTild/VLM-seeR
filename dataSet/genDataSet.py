from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.transforms.transforms import ComplexTo2D
import os
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator

root = "./classifier_example"
os.makedirs(root, exist_ok=True)
os.makedirs(root + "/train", exist_ok=True)
os.makedirs(root + "/val", exist_ok=True)
os.makedirs(root + "/test", exist_ok=True)
fft_size = 256
num_iq_samples_dataset = fft_size ** 2
class_list = TorchSigSignalLists.all_signals
family_list = TorchSigSignalLists.family_list
num_classes = len(class_list)
num_samples_train = len(class_list) * 5  # roughly 5 samples per class
num_samples_val = len(class_list) * 2
impairment_level = 0
# IQ-based mod-rec only operates on 1 signal
num_signals_max = 1
num_signals_min = 1

# ComplexTo2D turns a IQ array of complex values into a 2D array, with one channel for the real component, while the other is for the imaginary component
transforms = [ComplexTo2D()]

dataset_metadata = {
	"num_iq_samples_dataset": num_iq_samples_dataset,
	"fft_size": fft_size,
	"fft_stride": fft_size,
	"num_signals_max": num_signals_max,
	"num_signals_min": num_signals_min,
	"noise_power_db": 1,
	"signal_center_freq_min": 1000,
	"signal_center_freq_max": 2000,
	"sample_rate": 10000,
	"frequency_min": 1000,
	"frequency_max": 2000,
	"cochannel_overlap_probability": 0.2,
	"signal_duration_in_samples_min": 2000,
	"signal_duration_in_samples_max": 8000,
	"bandwidth_min": 1000,
	"bandwidth_max": 2000,
	"snr_db_min":10,
	"snr_db_max":20,
}


train_dataset = TorchSigIterableDataset(
	metadata=dataset_metadata,
	transforms=transforms,
	target_labels=None,
	signal_generators="all",
)
val_dataset = TorchSigIterableDataset(
	metadata=dataset_metadata, transforms=transforms, target_labels=None
)

class_list = train_dataset.class_names

train_dataloader = WorkerSeedingDataLoader(
	train_dataset, batch_size=4, collate_fn=lambda x: x
)
val_dataloader = WorkerSeedingDataLoader(val_dataset, collate_fn=lambda x: x)

# print(f"Data shape: {data.shape}")
# print(f"Targets: {targets}")
# next(train_dataset)

dc = DatasetCreator(
	dataloader=train_dataloader,
	root=f"{root}/train",
	overwrite=True,
	dataset_length=num_samples_train,

)
dc.create()

dc = DatasetCreator(
	dataloader=val_dataloader,
	root=f"{root}/val",
	overwrite=True,
	dataset_length=num_samples_val,
)
dc.create()

# train_dataset = StaticTorchSigDataset(
# 	root=f"{root}/train", target_labels=["class_index"]
# )
# val_dataset = StaticTorchSigDataset(root=f"{root}/val", target_labels=["class_index"])
#
# train_dataloader = WorkerSeedingDataLoader(train_dataset, batch_size=4)
# val_dataloader = WorkerSeedingDataLoader(val_dataset)
#
# print(train_dataset[3])
#
# next(iter(train_dataloader))