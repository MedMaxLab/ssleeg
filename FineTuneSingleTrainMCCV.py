# This code runs a single training of a Monte Carlo
# cross-validation using all the selected datasets

# set save subfolder (it can be modified for additional custom analyses)
flag_dir = ""

# ===========================
#  Section 1: package import
# ===========================
# This section includes all the packages to import.
import argparse
import glob
import os
import random
import pickle
import copy
import warnings
warnings.filterwarnings("ignore", message = "Using padding='same'", category = UserWarning)

# IMPORT STANDARD PACKAGES
import numpy as np
import pandas as pd

# IMPORT TORCH
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# IMPORT SELFEEG 
import selfeeg
import selfeeg.models as zoo
import selfeeg.dataloading as dl
import selfeeg.augmentation as aug
from selfeeg.ssl import fine_tune as train_model

# IMPORT REPOSITORY FUNCTIONS
import AllFnc
from AllFnc import split
from AllFnc.models import TransformEEG
from AllFnc.training import (
    loadEEG,
    lossBinary,
    lossMulti,
    get_performances,
    set_augmenter,
)

from AllFnc.pretraining import VICReg

from AllFnc.utilities import (
    restricted_float,
    positive_float,
    positive_int_nozero,
    positive_int,
    str2bool,
    str2list,
    str2listints,
    str_or_none,
    get_aug_idx
)

import warnings
warnings.filterwarnings(
    "ignore",
    message= "numpy.core.numeric is deprecated",
    category=DeprecationWarning
)

def _reset_seed_number(seed):
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-D",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = 1,
        required  = True,
    )
    parser.add_argument(
        "-S",
        "--sslmodel",
        dest      = "pretrainPath",
        metavar   = "pretrained model path",
        type      = str_or_none,
        nargs     = 1,
        required  = False,
        default   = None,
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        dest      = "pipelineToEval",
        metavar   = "preprocessing pipeline",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'ica',
    )
    parser.add_argument(
        "-t",
        "--task",
        dest      = "taskDataset",
        metavar   = "task",
        type      = str2list,
        nargs     = '?',
        required  = False,
        default   = ["ds003490"],
    )
    parser.add_argument(
        "-m",
        "--model",
        dest      = "modelToEval",
        metavar   = "model",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'transformeeg',
    )
    parser.add_argument(
        "-f",
        "--fold",
        dest      = "fold",
        metavar   = "fold",
        type      = int,
        nargs     = '?',
        required  = False,
        default   = 1,
        choices   = range(1,201),
    )
    parser.add_argument(
        "-c",
        "--cvsplitsize",
        dest      = "splitSize",
        metavar   = "cross validation split size",
        type      = str2listints,
        nargs     = '?',
        required  = False,
        default   = [0.64, 0.16, 0.2],
    )
    parser.add_argument(
        "-z",
        "--zscore",
        dest      = "zscore", 
        metavar   = "zscore",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
    )
    parser.add_argument(
        "-r",
        "--reminterp",
        dest      = "removeInterpolated",
        metavar   = "remove interpolated",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
    )
    parser.add_argument(
        "-b",
        "--batch",
        dest      = "batchsize",
        metavar   = "batch size",
        type      = positive_int_nozero,
        nargs     = '?',
        required  = False,
        default   = 64,
    )
    parser.add_argument(
        "-o",
        "--overlap",
        dest      = "overlap",
        metavar   = "windows overlap",
        type      = restricted_float,
        nargs     = '?',
        required  = False,
        default   = 0.25,
    )
    parser.add_argument(
        "-w",
        "--window",
        dest      = "window",
        metavar   = "window",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 16.0,
    )
    parser.add_argument(
        "-l",
        "--learningrate",
        dest      = "lr",
        metavar   = "learning rate",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 2.5e-4,
    )
    parser.add_argument(
        "-a",
        "--aug",
        dest      = "augmentation",
        metavar   = "Augmentation list",
        type      = str2list,
        nargs     = '?',
        required  = False,
        default   = None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest      = "verbose",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest      = "gpu",
        metavar   = "gpu device",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'cpu',
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest      = "seed",
        metavar   = "seed",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 42,
    )
    args = vars(parser.parse_args())
    
    if args['verbose']:
        print('running training with the following parameters:')
        print(' ')
        for key in args:
            if key == 'dataPath':
                print( f"{key:15} ==> {args[key][0]:<15}" ) 
            elif key == 'pretrainPath':
                try:
                    print( f"{key:15} ==> {args[key][0]:<15}" )
                except Exception:
                    print(f"{key:15} ==> ", args[key])
            elif key in ["augmentation", "taskDataset", "splitSize"]:
                print(f"{key:15} ==> ", args[key])
            else:
                print( f"{key:15} ==> {args[key]:<15}") 
    
    dataPath       = args['dataPath'][0]
    pretrainPath   = args['pretrainPath'][0]
    pipelineToEval = args['pipelineToEval']
    taskToEval     = args['taskDataset']
    modelToEval    = args['modelToEval'].casefold() 
    foldToEval     = args['fold'] - 1
    train_size     = args['splitSize'][0]
    val_size       = args['splitSize'][1]
    test_size      = args['splitSize'][2]
    z_score        = args['zscore']
    rem_interp     = args['removeInterpolated']
    batchsize      = args['batchsize']
    overlap        = args['overlap']
    window         = args['window']
    verbose        = args['verbose']
    lr             = args['lr']
    augment_list   = args['augmentation']
    device         = args['gpu'].casefold() 
    seed           = args['seed']
    downsample     = True
    equi           = False

    # Force as much determinism as possible, especially for transformeeg
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True

    # Define the device to use
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # ==================================
    #  Section 3: create partition list
    # ==================================
    datasetIDs = []
    train_id, val_id, test_id, exclude_id = {}, {}, {}, {}
    
    # ds003490 - ID 5 - 3Stim
    if "ds003490" in taskToEval:
        ctl_id_5 = [i for i in range(28,51)] + [3,5]
        pds_id_5 = [i for i in range(6,28)] + [1,2,4]
        part_c = split.create_repeated_holdout_split(
            ctl_id_5, 200, train_size, val_size, test_size, equi, 1, seed)
        part_p = split.create_repeated_holdout_split(
            pds_id_5, 200, train_size, val_size, test_size, equi, 1, seed)
        partition_list_1 = split.merge_partition_lists_noshift(part_c, part_p)
        datasetIDs.append("5")
        train_id[5] = partition_list_1[foldToEval][0]
        val_id[5]   = partition_list_1[foldToEval][1]
        test_id[5]  = partition_list_1[foldToEval][2]
        exclude_id[5] = list(
            set(ctl_id_5 + pds_id_5).difference(
                set(train_id[5] + val_id[5] + test_id[5])
            )
        )


    # ds002778 - ID 8 - UCSD
    if "ds002778" in taskToEval:
        ctl_id_8 = [i for i in range(1, 17)]
        pds_id_8 = [i for i in range(17,32)]
        if pipelineToEval != 'ica':
            ctl_id_8 = [1, 2, 4, 7,  8, 10, 17, 19, 20, 23, 24, 27, 28, 29, 30, 31]
            pds_id_8 = [3, 5, 6, 9, 11, 12, 13, 14, 15, 16, 18, 21, 22, 25, 26]
        part_c = split.create_repeated_holdout_split(
            ctl_id_8, 200, train_size, val_size, test_size, equi, 1, seed)
        part_p = split.create_repeated_holdout_split(
            pds_id_8, 200, train_size, val_size, test_size, equi, 1, seed)
        partition_list_2 = split.merge_partition_lists_noshift(part_c, part_p)
        datasetIDs.append("8")
        train_id[8] = partition_list_2[foldToEval][0]
        val_id[8]   = partition_list_2[foldToEval][1]
        test_id[8]  = partition_list_2[foldToEval][2]
        exclude_id[8] = list(
            set(ctl_id_8 + pds_id_8).difference(
                set(train_id[8] + val_id[8] + test_id[8])
            )
        )

    # ds004584 - ID 19 - PDEO
    if "ds004584" in taskToEval:
        pds_id_19 = [i for i in range(1, 101)]
        ctl_id_19 = [i for i in range(101, 141)]
        part_c = split.create_repeated_holdout_split(
            ctl_id_19, 200, train_size, val_size, test_size, equi, 1, seed)
        part_p = split.create_repeated_holdout_split(
            pds_id_19, 200, train_size, val_size, test_size, equi, 1, seed)
        partition_list_3 = split.merge_partition_lists_noshift(part_c, part_p)
        datasetIDs.append("19")
        train_id[19] = partition_list_3[foldToEval][0]
        val_id[19]   = partition_list_3[foldToEval][1]
        test_id[19]  = partition_list_3[foldToEval][2]
        exclude_id[19] = list(
            set(ctl_id_19 + pds_id_19).difference(
                set(train_id[19] + val_id[19] + test_id[19])
            )
        )

    # if empty, convert to None to avoid errors
    if len(exclude_id.keys())==0:
        exclude_id = None

    # EEGs are preprocessed with BIDSAlign and converted in pickle for faster loading
    glob_input = [i + "_*.pickle" for i in datasetIDs]
           
    # ======================================
    # Section 4: set the training parameters
    # =====================================
    if dataPath[-1] != os.sep:
        dataPath += os.sep
    if pipelineToEval[-1] != os.sep:
        eegpath = dataPath + pipelineToEval + os.sep
    else:
        eegpath = dataPath + pipelineToEval
    
    # Define the number of Channels to use. 
    if rem_interp:
        Chan = 32
    else:
        Chan = 61
    
    freq = 125 
    nb_classes = 2 # Define the number of classes to predict.
    Samples = int(freq*window) # For selfEEG's models instantiation
    
    # Set the class label in case of plot of functions
    class_labels = ['CTL', 'PD']
    
    # =====================================================
    #  Section 5: Define pytorch's Datasets and dataloaders
    # =====================================================
    
    # GetEEGPartitionNumber doesn't need the labels
    loadEEG_args = {
        'return_label': False, 
        'downsample': downsample, 
        'use_only_original': rem_interp,
        'apply_zscore': z_score
    }

    # calculate dataset length.
    EEGlen = dl.get_eeg_partition_number(
        eegpath, freq, window, overlap, 
        file_format             = glob_input,
        load_function           = loadEEG,
        optional_load_fun_args  = loadEEG_args,
        includePartial          = False if overlap == 0 else True,
        verbose                 = verbose
    )
    
    # Now we also need to load the labels
    loadEEG_args['return_label'] = True
    
    # Set functions to retrieve dataset, subject, and session from each filename.
    # They will be used by GetEEGSplitTable to perform a subject based split
    dataset_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[0])
    subject_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[1])

    # Create training, validation, and test sets according
    # to the datasets selected and the current MCCV fold
    EEGsplit= dl.get_eeg_split_table(
        partition_table      = EEGlen,
        exclude_data_id      = exclude_id,
        val_data_id          = val_id,
        test_data_id         = test_id, 
        split_tolerance      = 0.001,
        dataset_id_extractor = dataset_id_ex,
        subject_id_extractor = subject_id_ex,
        perseverance         = 10000
    )
    
    if verbose:
        print(' ')
        print('Subjects used for test')
        print(test_id)
        print('Subjects excluded in this run')
        print(exclude_id)
    
    # Define Datasets and preload all data
    trainset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'train', 
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    trainset.preload_dataset()
    
    valset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'validation',
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args,
    )
    valset.preload_dataset()
    
    testset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'test',
        supervised             = True,
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args,
    )
    testset.preload_dataset()
    
    trainset.x_preload = trainset.x_preload.to(device=device)
    trainset.y_preload = trainset.y_preload.to(device=device)
    valset.x_preload = valset.x_preload.to(device=device)
    valset.y_preload = valset.y_preload.to(device=device)
    testset.x_preload = testset.x_preload.to(device=device)
    testset.y_preload = testset.y_preload.to(device=device)

    if verbose:
        print("training, validation, test set sizes")
        print(trainset.x_preload.shape, valset.x_preload.shape, testset.x_preload.shape)
    
    # Finally, Define Dataloaders
    trainloader = DataLoader( dataset=trainset, batch_size=batchsize, shuffle=True)
    valloader   = DataLoader( dataset=valset,   batch_size=batchsize, shuffle=False)
    testloader  = DataLoader( dataset=testset,  batch_size=batchsize, shuffle=False)
    
    # ===================================================
    #  Section 6: define the loss, model, and optimizer
    # ==================================================
    
    lossVal = None
    validation_loss_args = []
    lossFnc = lossBinary
    if 'ds004584' in taskToEval:
        from sklearn.utils.class_weight import compute_class_weight
        y_numpy = trainset.y_preload.cpu().numpy()
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_numpy), y = y_numpy
        )
        class_weights = torch.from_numpy(class_weights)
        class_weights = class_weights.to(dtype=torch.float32, device=device)
        def lossBinary2 (yhat, ytrue, class_weights=class_weights):
            true_weights = torch.zeros_like(ytrue)
            true_weights[ytrue==0] = class_weights[0]
            true_weights[ytrue==1] = class_weights[1]
            return torch.nn.functional.binary_cross_entropy_with_logits(
                yhat.flatten(), ytrue, weight=true_weights)
        lossFnc = lossBinary2

    # Set data augmentation
    if augment_list is None:
        augmenter = None
    else:
        augmenter = set_augmenter(augment_list, fs=freq, winlen=window)
        augidx1 = get_aug_idx(augment_list[0])
        augidx2 = get_aug_idx(augment_list[1])

    
    # define model
    Mdl = TransformEEG(nb_classes, Chan, Chan*4, False, -31*5, seed)
    
    if pretrainPath is not None:
        if verbose:
            print('loading pretrained weights')

        import copy
        from collections import OrderedDict
        def get_encoder(model, device="cpu", as_ordered_dict=False):
            if as_ordered_dict:
                enc = OrderedDict(
                    [(k, v.to(device=device, copy=True)) for k, v in model.encoder.state_dict().items()]
                )
            else:
                enc = copy.deepcopy(self.encoder).to(device=device)
            return enc
        mdl_siamese = VICReg(Mdl.encoder, [128, 128])
        mdl_siamese.load_state_dict(torch.load(pretrainPath, weights_only=True))
        Mdl.encoder.load_state_dict(get_encoder(mdl_siamese, as_ordered_dict=True))

    
    Mdl.to(device = device)
    Mdl.train()
    if verbose:
        print(' ')
        ParamTab = selfeeg.utils.count_parameters(Mdl, False, True, True)
        print(' ')

    
    gamma = 0.99
    optimizer = torch.optim.Adam( Mdl.parameters(), betas = (0.75, 0.999), lr = lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    
    # Define selfEEG's EarlyStopper
    earlystop = selfeeg.ssl.EarlyStopping(
        patience  = 20, 
        min_delta = 1e-04, 
        record_best_weights = True
    )
    
    # =============================
    #  Section 7: train the model
    # =============================
    _reset_seed_number(seed)
    loss_summary = train_model(
        model                 = Mdl,
        train_dataloader      = trainloader,
        epochs                = 300,
        optimizer             = optimizer,
        loss_func             = lossFnc,
        augmenter             = augmenter,
        lr_scheduler          = scheduler,
        EarlyStopper          = earlystop,
        validation_dataloader = valloader,
        validation_loss_func  = lossVal,
        validation_loss_args  = validation_loss_args,
        verbose               = verbose,
        device                = device,
        return_loss_info      = True
    )
    
    # ===============================
    #  Section 8: evaluate the model
    # ===============================
    scores = {}
    scores['loss_progression'] = loss_summary
    earlystop.restore_best_weights(Mdl)
    Mdl.to(device=device)
    Mdl.eval()

    # Evaluate model on the window level with the standard 0.5 threshold
    scores['metrics'] = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels,
        th             = 0.5,
    )
    # if not verbose:
    bal_acc = scores['metrics']['accuracy_weighted']
    if not verbose:
        print(f'Balanced accuracy on windows with threshold 0.500 --> {bal_acc:.5f}')

    # ==================================
    #  Section 9: Save model and metrics
    # ==================================

    # we will create a custom name summarizing
    # all the important parameters using for this training
    start_piece_mdl = 'Results/Finetuning/Models/'
    start_piece_res = 'Results/Finetuning/Results/'

    # To redirect supplementary analysis
    if (flag_dir is not None) and (flag_dir != ""):
        if flag_dir[-1] != os.sep:
            flag_dir += os.sep
        start_piece_mdl += flag_dir
        start_piece_res += flag_dir

    # Select which subdirectory within Results and Models should be used
    if 'ds003490' in taskToEval:
        dir_piece += 'ds003490'
    elif 'ds002778' in taskToEval:
        dir_piece += 'ds002778'
    elif 'ds004584' in taskToEval:
        dir_piece += 'ds004584'
    dir_piece += os.sep

    start_piece_mdl += dir_piece
    start_piece_res += dir_piece

    
    freq_piece = '125'
    task_piece = 'pds'
    mdl_piece = 'etr'
    pipe_piece = 'ica'

    if augment_list is None:
        aug1_piece = '000'
        aug2_piece = '000'
    else:
        aug1_piece = str(augidx1+1).zfill(3)
        aug2_piece = str(augidx2+1).zfill(3)

    
    if pretrainPath is not None:
        pretrain_name = pretrainPath.split(os.sep)[-1]
        ep_idx = pretrain_name.index('ep')
        pretr_piece = pretrain_name[ep_idx-3:ep_idx]
        if pretr_piece[0] == '_':
            pretr_piece = pretr_piece[1:].zfill(3)
    else:
        pretr_piece = '000'
        
    
    fold_piece = str(foldToEval+1).zfill(3)
    over_piece = str(int(overlap*100)).zfill(3)
    chan_piece = str(Chan).zfill(3)
    win_piece = str(round(window)).zfill(3)
    size_piece = str(int((train_size+val_size)*100)).zfill(3)
    
    file_name = '_'.join(
        [task_piece, pipe_piece, freq_piece, mdl_piece, fold_piece, chan_piece, 
         win_piece, over_piece, size_piece, pretr_piece, aug1_piece, aug2_piece
        ]
    )
    model_path = start_piece_mdl + file_name + '.pt'
    results_path = start_piece_res + file_name + '.pickle'

    verbose = True
    if verbose:
        print('saving model and results in the following paths')
        print(model_path)
        print(results_path)
    
    # Save the scores
    with open(results_path, 'wb') as handle:
        pickle.dump(scores, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print('run complete')
