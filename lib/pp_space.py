"""
Post-processing and analysis algorithm for InfoVAE in latent space and physical space

Author: @alsolra
Editing: @yuningw
Adapted for InfoVAE
"""

import torch
import numpy as np
import h5py
from lib.init import pathsBib

################################
### Main programme for spatial analysis
###############################
def spatial_Mode(fname,
                 model,
                 latent_dim,
                 train_data,
                 test_data,
                 dataset_train,
                 dataset_test,
                 mean, std,
                 device,
                 if_order=True,
                 if_nlmode=True,
                 if_Ecumt=True,
                 if_Ek_t=True):
    """
    Main function for spatial mode analysis and dataset generation.
    """

    print(f"INFO: Start spatial mode generating")
    if if_order:
        order, Ecum = get_order(model, latent_dim,
                                train_data,
                                dataset_train,
                                std, device)
        print(f"INFO: RANKING DONE")
    else:
        order = None
        Ecum = None

    if if_nlmode:
        NLvalues, NLmodes = getNLmodes(model, order[0], latent_dim, device)
        print("INFO: Non-linear mode generated")
    else:
        NLmodes = None
        NLvalues = None

    if if_Ecumt:
        Ecum_test = get_EcumTest(model, latent_dim, test_data, dataset_test, std, device, order)
        print('INFO: Test E_cum generated')
    else:
        Ecum_test = None

    if if_Ek_t:
        Ek_t = get_Ek_t(model=model, data=test_data, device=device)
    else:
        Ek_t = None

    is_save = createModesFile(fname, model, latent_dim,
                              dataset_train, dataset_test,
                              mean, std, device,
                              order, Ecum, Ecum_test,
                              NLvalues, NLmodes, Ek_t)

    if is_save: print("INFO: Successfully DONE!")

    return is_save

################################
### Basic function for using InfoVAE
###############################

def encode(model, data, device):
    """
    Encode flow field into latent space using the encoder.
    Returns mean and logvariance.
    """
    mean_list = []
    logvar_list = []
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device, non_blocking=True)
            out = model.encoder(batch)
            mean, logvariance = torch.chunk(out, 2, dim=1)
            mean_list.append(mean.cpu().numpy())
            logvar_list.append(logvariance.cpu().numpy())

    means = np.concatenate(mean_list, axis=0)
    logvars = np.concatenate(logvar_list, axis=0)
    return means, logvars
"""
def decode(model, data, device):
 
    #Decode latent vectors back to physical space.



    dataset = torch.utils.data.DataLoader(dataset=torch.from_numpy(data),
                                          batch_size=512,
                                          shuffle=False,
                                          num_workers=2)
    rec_list = []
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device)
            rec = model.decoder(batch)
            rec_list.append(rec.cpu().numpy())
    return np.concatenate(rec_list, axis=0)
"""

def decode(model, data, device):

    print("\n-----------------------------")
    print("DEBUG decode")
    print("Input latent shape :", data.shape)
    print("-----------------------------") 

    dataset = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(data),
        batch_size=512,
        shuffle=False,
        num_workers=1)

    rec_list = []

    with torch.no_grad():

        for k, batch in enumerate(dataset):

            if k == 0:
                print("First batch shape :", batch.shape)

            batch = batch.to(device)

            rec = model.decoder(batch)

            if k == 0:
                print("Decoder output :", rec.shape)

            rec_list.append(rec.cpu().numpy())

    print("Number of batches :", len(rec_list))

    return np.concatenate(rec_list, axis=0)

def get_Ek_stream(model, original, latent, std, device, batch_size=512):
    """
    Compute reconstructed kinetic energy without storing the whole reconstruction.
    """

    loader = torch.utils.data.DataLoader(
        dataset=torch.from_numpy(latent),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    numerator = 0.0

    denominator = np.sum(
        original[:,0]**2 +
        original[:,1]**2
    )

    index = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            rec = model.decoder(batch).cpu().numpy()

            rec *= std

            end = index + rec.shape[0]

            ref = original[index:end]

            numerator += np.sum(
                (ref[:,0]-rec[:,0])**2 +
                (ref[:,1]-rec[:,1])**2
            )

            index = end

    return 1.0 - numerator/denominator
    
def get_samples(model, dataset_train, dataset_test, device):
    """
    Quickly obtain reconstructed snapshots for testing or visualization.
    """
    with torch.no_grad():
        if dataset_train is not None:
            for batch_train in dataset_train:
                batch_train = batch_train.to(device, non_blocking=True)
                rec_train = model(batch_train)[0]  # only reconstruction
                rec_train = rec_train.cpu().numpy()[-1]
                true_train = batch_train.cpu().numpy()[-1]
                break
        else:
            rec_train, true_train = None, None

        if dataset_test is not None:
            for batch_test in dataset_test:
                batch_test = batch_test.to(device, non_blocking=True)
                rec_test = model(batch_test)[0]  # only reconstruction
                rec_test = rec_test.cpu().numpy()[-1]
                true_test = batch_test.cpu().numpy()[-1]
                break
        else:
            rec_test, true_test = None, None

        return rec_train, rec_test, true_train, true_test

################################
### Spatial-mode generate and analysis
###############################

def calcmode(model, latent_dim, mode, device):
    """
    Generate non-linear mode with unit vector.
    """
    z_sample = np.zeros((1, latent_dim), dtype=np.float32)
    z_sample[:, mode] = 1
    with torch.no_grad():
        mode_out = model.decoder(torch.from_numpy(z_sample).to(device)).cpu().numpy()
    return mode_out

def get_spatial_modes(model, latent_dim, device):
    """
    Obtain spatial modes from decoder.
    """
    with torch.no_grad():
        zero_output = model.decoder(torch.from_numpy(np.zeros((1, latent_dim), dtype=np.float32)).to(device)).cpu().numpy()
    modes = np.zeros((latent_dim, zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]))
    for mode in range(latent_dim):
        modes[mode, :, :, :] = calcmode(model, latent_dim, mode, device)
    return zero_output, modes

def getNLmodes(model, mode, latent_dim, device):
    """
    Obtain single non-linear spatial mode.
    """
    zero_output = decode(model, np.zeros((1, latent_dim), dtype=np.float32), device)
    NLvalues = np.arange(-2, 2.1, .1)
    NLmodes = np.zeros((NLvalues.shape[0], zero_output.shape[1], zero_output.shape[2], zero_output.shape[3]), dtype=np.float32)
    for idx, value in enumerate(NLvalues):
        latent = np.zeros((1, latent_dim), dtype=np.float32)
        latent[0, mode] = value
        NLmodes[idx, :, :, :] = decode(model, latent, device)
    return NLvalues, NLmodes

def get_order(model, latent_dim, data, dataset, std, device):
    """
    Rank spatial modes based on cumulative energy.
    """
    print('#'*30)
    print('Ordering modes')
    modes, _ = encode(model, dataset, device)
    
    print("="*60)
    print("DEBUG get_order")
    print("modes.shape      :", modes.shape)
    print("data.shape       :", data.shape)
    print("latent_dim       :", latent_dim)
    print("std.shape        :", std.shape)
    print("="*60)
    
    u = data[:, :, :, :] * std[:, :, :, :]
    m = np.zeros(latent_dim, dtype=int)
    n = np.arange(latent_dim)
    Ecum = []
    partialModes = np.zeros_like(modes, dtype=np.float32)

    for i in range(latent_dim):
        Eks = []
        print(f"\n===== OUTER LOOP i = {i} =====")

        for j in n:
            partialModes *= 0
            partialModes[:, m[:i]] = modes[:, m[:i]]
            partialModes[:, j] = modes[:, j]
            print(f"i={i}, j={j}")
            print("partialModes.shape =", partialModes.shape)
            """
            u_pred = decode(model, partialModes, device) * std
            Eks.append(get_Ek(u, u_pred))
            """
            Ek = get_Ek_stream(
            model=model,
            original=u,
            latent=partialModes,
            std=std,
            device=device
            )

            Eks.append(Ek)
        Eks = np.array(Eks).squeeze()
        ind = n[np.argmax(Eks)]
        m[i] = ind
        n = np.delete(n, np.argmax(Eks))
        Ecum.append(np.max(Eks))
    Ecum = np.array(Ecum)
    return np.array(m), Ecum

################################
### Assessment on Energy
###############################

def get_Ek(original, rec):
    """
    Calculate energy percentage reconstructed.
    """
    TKE_real = original[:, 0, :, :] ** 2 + original[:, 1, :, :] ** 2
    u_rec = rec[:, 0, :, :]
    v_rec = rec[:, 1, :, :]
    return 1 - np.sum((original[:, 0, :, :] - u_rec) ** 2 + (original[:, 1, :, :] - v_rec) ** 2) / np.sum(TKE_real)

def get_Ek_t(model, data, device):
    """
    Reconstructed energy for snapshots (InfoVAE compatible).
    """
    dataloader = torch.utils.data.DataLoader(dataset=torch.from_numpy(data),
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1)
    rec_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            rec_list.append(out[0].cpu().numpy())  # only reconstruction
    rec = np.concatenate(rec_list, axis=0)
    Ek_t = np.zeros((rec.shape[0]))
    for i in range(rec.shape[0]):
        Ek_t[i] = get_Ek(data[np.newaxis, i], rec[np.newaxis, i])
    return Ek_t

def get_EcumTest(model, latent_dim, data, dataset, std, device, order):
    """
    Accumulative energy of test database.
    """
    modes, _ = encode(model, dataset, device)
    u = data[:, :, :, :] * std[:, :, :, :]
    Ecum = []
    for i in range(latent_dim):
        partialModes = np.zeros_like(modes, dtype=np.float32)
        partialModes[:, order[:i+1]] = modes[:, order[:i+1]]
        """
        u_pred = decode(model, partialModes, device)
        u_pred *= std[:, :, :, :]
        Ecum.append(get_Ek(u, u_pred))
        """
        Ek = get_Ek_stream(
        model=model,
        original=u,
        latent=partialModes,
        std=std,
        device=device
        )

        Ecum.append(Ek)
    return np.array(Ecum)

################################
### I/O
###############################
def createModesFile(fname,
                    model,
                    latent_dim,
                    dataset_train, dataset_test,
                    mean, std,
                    device,
                    order,
                    Ecum, Ecum_test,
                    NLvalues, NLmodes,
                    Ek_t):
    """
    Save post-processed InfoVAE results.
    """
    if_save = False
    means_train, stds_train = encode(model, dataset_train, device)
    means_test, stds_test = encode(model, dataset_test, device)
    zero_output, modes = get_spatial_modes(model, latent_dim, device)
    if order is None:
        order = np.arange(latent_dim)

    with h5py.File(fname + ".hdf5", 'w') as f:
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)
        f.create_dataset('vector', data=means_train)
        f.create_dataset('vector_test', data=means_test)
        f.create_dataset('stds_vector', data=stds_train)
        f.create_dataset('stds_vector_test', data=stds_test)
        f.create_dataset('modes', data=modes)
        f.create_dataset('zero_output', data=zero_output)
        f.create_dataset('NLvalues', data=NLvalues)
        f.create_dataset('NLmodes', data=NLmodes)
        f.create_dataset('order', data=order)
        f.create_dataset('Ecum', data=Ecum)
        f.create_dataset('Ecum_test', data=Ecum_test)
        f.create_dataset('Ek_t', data=Ek_t)
    with h5py.File(pathsBib.data_path + 'latent_data' + ".h5py", 'w') as f:
        f.create_dataset('vector', data=means_train)
        f.create_dataset('vector_test', data=means_test)

    if_save = True
    print(f"INFO: Post-processing results saved as dataset: {fname}")
    return if_save