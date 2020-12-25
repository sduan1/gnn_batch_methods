import torch
from torch.cuda import device_count
from torch_geometric.data import Data, DataLoader, DataListLoader

def model_init_test(model, size, num_classes):
    inputs = torch.rand(*size)
    try:
        out = model(inputs)
        assert out.size == (size[0], num_classes)
    except:
        print(f'Warning: Model failed')

def model_output_test(output, input_data):
    if output.size() != (input_data.x.size()[0], 2):
        print(f'Warning: output size error, expected {(input_data.x.size()[0], 2)} but got {output.size()} instead')

def dataset_test(dataset, verbose=1):
    if verbose==1:
        print(f'Loading dataset: {dataset}')
    #Verify dataset type and content
    if not isinstance(dataset, Data):
        print(f'Warning: Using non_pytorch_geometric dataset. The dataset has type{type(Data)}')
    if dataset.x is None or dataset.y is None:
        print(f'Warning: Dataset is incomplete')

def gpu_test(gpu_num):
    print(f'Using {device_count()} GPUs')
    if device_count()!=gpu_num:
        print('Warning GPU set up bug')

def data_list_test(datalist, verbose=1):
    if not isinstance(datalist, list):
        print(f'Warning: datalist has type {type(datalist)}')

    if verbose==1:
        print(f'Datalist info: len of datalist: {len(datalist)}')

