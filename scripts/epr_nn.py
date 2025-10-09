import os
import csv
import numpy as np
import torch
import torch.nn as nn
import mne
from mne.minimum_norm import read_inverse_operator
from mne.source_estimate import _prepare_label_extraction
from tqdm import tqdm

# Set the number of threads for PyTorch
torch.set_num_threads(4)

# Define a neural network block with residual connections
class DeepRitzBlock(nn.Module):
    def __init__(self, h_size):
        super(DeepRitzBlock, self).__init__()
        # Define a sequential block with two linear layers and Tanh activations
        self.block = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.Tanh(),
            nn.Linear(h_size, h_size),
            nn.Tanh()
        )

    def forward(self, x):
        # Apply the block and add the input (residual connection)
        return self.block(x) + x

# Define a neural network model using DeepRitzBlock
class NeuralNetwork(nn.Module):
    def __init__(self, in_size, h_size=10, block_size=1, dev="cpu"):
        super(NeuralNetwork, self).__init__()
        self.dev = dev
        self.dim_input = in_size
        self.dim_h = h_size
        self.num_blocks = block_size

        # Initialize the layers list with either a linear layer or padding
        layers = [nn.ConstantPad1d((0, self.dim_h - self.dim_input), 0) if self.dim_input <= self.dim_h else
                  nn.Linear(self.dim_input, self.dim_h)]

        # Append DeepRitzBlock instances to the layers list
        for _ in range(self.num_blocks):
            layers.append(DeepRitzBlock(self.dim_h))

        # Add a final linear layer to map back to the input size
        layers.append(nn.Linear(self.dim_h, self.dim_input))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the model
        return self.model(x)

def generate_surrogate(x):
    x_surrogate = []
    for i, x_i in enumerate(x):
        x_surrogate_ = []
        for l in range(len(x_i)):
            fft_original = np.fft.rfft(x_i[l])
            magnitude = np.abs(fft_original)
            random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(fft_original)))
            fft_random = magnitude * random_phases
            random_signal = np.fft.irfft(fft_random)
            x_surrogate_.append(random_signal)
        x_surrogate.append(np.array(x_surrogate_))
    return x_surrogate

def EPR_nn_est_params(data_, flip_vec, dt, epoch_max=3000, step_test=100, data_normalisation=True,
                      reverse_training=False, dim_h=10, num_blocks=2):
    data_ij = data_
    dim, length = data_ij.shape

    if reverse_training:
        xt = torch.Tensor(data_ij[:, length // 2:].T)
        xt_test = torch.Tensor(data_ij[:, :length // 2].T)
    else:
        xt = torch.Tensor(data_ij[:, :length // 2].T)
        xt_test = torch.Tensor(data_ij[:, length // 2:].T)

    dim_x = dim
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_network = NeuralNetwork(dim_x, dim_h, num_blocks, dev).to(dev)

    data = xt - torch.mean(xt, dim=0)
    if data_normalisation:
        data /= torch.std(data, dim=0)

    data_mid = 0.5 * (data[1:, :] + data[:-1, :])
    xdiff = data[1:, :] - data[:-1, :]

    optimizer = torch.optim.Adadelta(force_network.parameters(), lr=1e-2)

    All_loss_train = []
    for epoch in range(epoch_max):
        force_network.train()
        optimizer.zero_grad()
        dxmid = force_network(data_mid)
        jj = torch.sum(dxmid * xdiff, dim=1)
        loss = -2 * torch.mean(jj) ** 2 / (dt * torch.var(jj))
        loss.backward()
        optimizer.step()

        if epoch % step_test == 0:
            torch.save(force_network.state_dict(), f'saved_params/force_network_params_epoch_{epoch}.pt')
            All_loss_train.append(-loss.item())

    saved_params_dir = 'saved_params' # Create a folder with this name in the same folder as the code
    steps = range(0, epoch_max, step_test)
    test_losses = []

    data_test = xt_test - torch.mean(xt_test, dim=0)
    if data_normalisation:
        data_test /= torch.std(data_test, dim=0)

    data_mid_test = 0.5 * (data_test[1:, :] + data_test[:-1, :])
    xdiff_test = data_test[1:, :] - data_test[:-1, :]

    for step in steps:
        param_file = os.path.join(saved_params_dir, f'force_network_params_epoch_{step}.pt')
        if os.path.exists(param_file):
            force_network.load_state_dict(torch.load(param_file))
            force_network.eval()
            with torch.no_grad():
                dxmid_test = force_network(data_mid_test)
                jj_test = torch.sum(dxmid_test * xdiff_test, dim=1)
                loss_test = -2 * torch.mean(jj_test) ** 2 / (dt * torch.var(jj_test))
                test_losses.append(-loss_test.item())
        else:
            test_losses.append(None)

    epr_train, epr_test = np.max(All_loss_train), np.max(test_losses)
    print(f"Maximum EPR on training data: {epr_train}")
    print(f"Maximum EPR on test data: {epr_test}")

    return test_losses, All_loss_train

list_subjs = [] # List of subjects ID to fill

list_subj_ses = [[list_subjs[k], 'ses1'] for k in range(len(list_subjs))]
list_subj_ses.extend([[list_subjs[k], 'ses2'] for k in range(len(list_subjs))])

surrogate = True

subjects_dir = "path_to_the_folder_where_the_MEG_data_are" # To fill
data_dir = 'path_to_the_output_folder' # To fill
fs_dir = "path_to_the_freesurfer_folder" # To fill
parc = 'HCPMMP1_combined'
# Number of times we compute the EPR
n_samples = 1
for subject_ses in list_subj_ses:
    subject, ses = subject_ses
    inv_filename = os.path.join(subjects_dir, subject, f'{subject}-{ses}-inv.fif')
    stc_filename = os.path.join(subjects_dir, subject, f'{subject}-{ses}-dSPM-lh.stc')

    inv = read_inverse_operator(inv_filename)
    src = inv["src"]
    stc = mne.read_source_estimate(stc_filename)

    labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=fs_dir, sort=False)
    label_ts = mne.extract_label_time_course(stc, labels_parc, src, mode=None, allow_empty=True, mri_resolution=True)
    label_vertidx, src_flip = _prepare_label_extraction(stc, labels_parc, src, 'pca_flip', allow_empty=True, use_sparse=False)

    inds = [i for i, label in enumerate(labels_parc) if '???' not in label.name]
    dt = stc.tstep
    sigma_list = [[] for _ in inds]

    label_ts = [label_ts[k] for k in inds]
    if surrogate:
        label_ts = generate_surrogate(label_ts)

    label_vertidx = [label_vertidx[k] for k in inds]
    src_flip = [src_flip[k] for k in inds]

    del stc
    count = 0
    for region_ind in tqdm(range(len(inds)), desc=f"Processing brain region {count}"):
        count += 1
        for n_s in range(n_samples):
            data_ = label_ts[region_ind]
            if n_s < int(n_samples/2):
                test_losses, All_loss_train = EPR_nn_est_params(data_, src_flip[region_ind], dt,
                epoch_max=500, step_test=10, dim_h=20, num_blocks=1)
            else:
                test_losses, All_loss_train = EPR_nn_est_params(data_, src_flip[region_ind], dt,
                epoch_max=500, step_test=10, reverse_training=True, dim_h=20, num_blocks=1)
            sigma_list[region_ind].append([np.max(All_loss_train), np.max(test_losses)])

    # Construct the list of dictionaries
    mydict = [
        {
            'region': labels_parc[inds[i]].name,
            **{f'sigma_train{k}': sigma_list[i][k][0] for k in range(n_samples)},
            **{f'sigma_test{k}': sigma_list[i][k][1] for k in range(n_samples)}
        }
        for i in range(len(inds))
    ]
    fields = mydict[0].keys()
    mode = 'no_PCA_clean'
    output_dir = os.path.join(data_dir,"output", f'{subject}_output_{ses}')
    output_filename = os.path.join(output_dir, f'{subject}_{ses}_{parc}_{mode}_RNN-zscore.csv')

    if surrogate:
        output_filename = os.path.join(output_dir, f'{subject}_{ses}_{parc}_{mode}_RNN-zscore_surrogate.csv')

    with open(output_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mydict)

    print(f'{subject} {ses} done!')
