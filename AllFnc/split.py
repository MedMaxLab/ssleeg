from decimal import Decimal, ROUND_HALF_UP
import math
import numpy as np
import random

__all__ = [
    'create_nested_kfold_subject_split',
    'create_repeated_holdout_split',
    'merge_partition_lists',
    'merge_partition_lists_noshift',
]

def create_nested_kfold_subject_split(
    subj: int or list[int], 
    folds: int = 10, 
    folds_nested: int = 5, 
    start_with: int = 1,
    seed: int = 83136297, 
) -> list[list[int], list[int], list[int]]:
    '''
    create_nested_kfold_subject_split creates a partition list to run a
    "nested subject-based k-fold cross validation". 

    Parameters
    ----------
    subj: int or list[int]
        The number of subjects of the dataset. It can be a list with the specific
        subject ids to use during the data partition.
    folds: int, optional
        The number of outer folds. It must be a positive integer. 
        Default = 10
    folds_nested: int, optional
        The number of inner folds. It must be a positive integer with value
        bigger than floor(subj-subj/folds).
        Default = 5
    start_with: int, optional
        The starting ID to use when creating the list of subject IDs. It will
        be considered only if subj is given as an integer. For example, if subj 
        is 10 and start_with is 5, the list of IDs to partition is 
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14].
        Default = 1
    seed: int, optional
        The seed to use during data partition. This will make results reproducible.
        Default = 83136297

    Returns
    -------
    partition_list: list
        The partition list. Each element in the list is a collection 
        of three lists, representing a specific partition. 
        The first contains the subjects IDs included in the training set,
        the second the subjects IDs included in the validation set,
        the third the subjects IDs included in the test set.
        Note that the length of the partition_list is outer_folds*inner_folds,
        and the partition at index i refers to the 
        mod(i+1, inner_folds) inner fold of the ceil((i+1)/outer_folds) outer fold.
    
    '''
    # initialize list of subjects
    if isinstance(subj,list):
        subj_list = subj
        subj = len(subj_list)
    else:
        start_idx = start_with
        end_idx = start_idx + subj
        subj_list = [i for i in range(start_idx, end_idx)]
    
    if folds>subj:
        raise ValueError('Number of folds cannot be greater' +
                         ' than the number of subjects')
    if folds_nested > math.floor(subj-subj/folds):
        raise ValueError(
            '''
            Number of nested folds cannot be greater than the number of subjects 
            minus the number of subject in the test fold, so you must satisfy 
            ceil(subjects - subj/folds) > folds_nested
            ''' 
        )

    # to calculate list diffs by preserving order we will use this lambda function
    list_diff = lambda l1,l2: [x for x in l1 if x not in l2]
    
    # preallocate list of split. Each split is represented by three list:
    # the first for train, the second for validation, the third for test
    partition_list = [[[None],[None],[None]] for i in range(folds*folds_nested)]

    # shuffle list according to given seed
    random.seed(seed)
    random.shuffle(subj_list)
    subj_list_set = set(subj_list)
    
    # initialize test counter to provide a simple way 
    # to create splits with a similar size
    prt_cnt = 0
    tst_cnt = 0
    for i in range(folds):
        
        # each nested fold share the same test set so we initialize it outside the 
        # second loop the best way to create test sets is to slide along the 
        # shuffled list, then get the remaining subject for the next for loop
        tst_to_add = Decimal((subj-tst_cnt)/(folds-i))
        tst_to_add = int(tst_to_add.to_integral_value(rounding=ROUND_HALF_UP))
        tst_id = subj_list[tst_cnt:(tst_cnt+tst_to_add)]
        tst_cnt += tst_to_add

        # extract list of remaining subjects for the creation of the train/val split
        val_cnt = 0
        subj_left = list_diff(subj_list,tst_id)
        subj_left_set = set(subj_left)
        Nleft = len(subj_left)
        for k in range(folds_nested):

            # we need to create a train and validation from the remaining subjects 
            # the best way to do it is to create the validation by sliding along 
            # the list of remaining subjects
            val_to_add = Decimal((Nleft-val_cnt)/(folds_nested-k))
            val_to_add = int(val_to_add.to_integral_value(rounding=ROUND_HALF_UP))
            val_id = subj_left[val_cnt:(val_cnt+val_to_add)]
            val_cnt += val_to_add

            # finally we create the last
            trn_id = subj_left_set.difference(set(val_id))

            #assign everything
            partition_list[prt_cnt][0]=list(trn_id)
            partition_list[prt_cnt][1]=list(val_id)
            partition_list[prt_cnt][2]=list(tst_id)
            prt_cnt += 1

    return partition_list


def create_repeated_holdout_split(
    subj: int or list[int], 
    folds: int = 100, 
    train_size: float = 0.64,
    val_size: float = 0.16,
    test_size: float = 0.2,
    homogeneous_sampling: bool = True,
    start_with: int = 1,
    seed: int = None, 
) -> list[list[int], list[int], list[int]]:
    '''
    create_repeated_holdout_split creates a set of partitions
    using the repeated hold-out approach.

    Parameters
    ----------
    subj: int or list[int]
        The number of subjects of the dataset. It can be a list with the specific
        subject ids to use during the data partition.
    folds: int, optional
        The number of outer folds. It must be a positive integer. 
        Default = 10
    train_size: float, optional
        The percentage of subjects to put in the training set.
        Must be a positive number in (0,1).
        Default = 0.64
    val_size: float, optional
        The percentage of subjects to put in the validation set.
        Must be a positive number in (0,1).
        Default = 0.16
    test_size: float, optional
        The percentage of subjects to put in the test set.
        Must be a positive number in (0,1).
        Default = 0.40
    homogeneous_sampling: bool, optional
        If True, try to sample the subjects uniformly between partitions and splits.
        In this case, the function will try to return a set of partitions where the
        number of times a subjects appear in the train/validation/test set
        across all splits is almost the same.
        Default = True
    start_with: int, optional
        The starting ID to use when creating the list of subject IDs. It will
        be considered only if subj is given as an integer. For example, if subj 
        is 10 and start_with is 5, the list of IDs to partition is 
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14].
        Default = 1
    seed: int, optional
        The seed to use during data partition. This will make results reproducible.
        Default = None

    Returns
    -------
    partition_list: list
        The partition list. Each element in the list is a collection 
        of three lists, representing a specific partition. 
        The first contains the subjects IDs included in the training set,
        the second the subjects IDs included in the validation set,
        the third the subjects IDs included in the test set.

    Example
    -------
    >>> Nsubj = 31
    >>> CVList = create_repeated_holdout_split(subj= Nsubj, folds = 100, seed = 42)

    check if there are repeated subjects between sets of the same splits

    >>> for i in range(len(CVList)):
    >>>     train = set(CVList[i][0])
    >>>     val = set(CVList[i][1])
    >>>     test = set(CVList[i][2])
    >>>     if len(train.intersection(val)) != 0:
    >>>         raise ValueError("error train-val at", i)
    >>>     if len(train.intersection(test)) != 0:
    >>>         raise ValueError("error train-test at", i)
    >>>     if len(val.intersection(test)) !=0:
    >>>         raise ValueError("error val-test at", i)

    check the subject count between sets across all splits
    
    >>> cnt = np.zeros(Nsubj+1) # it start from 1 as default, not 0
    >>> for i in range(len(CVList)):
    >>>     cnt[CVList[i][0]] += 1
    >>> print(cnt[1:])
    >>> cnt = np.zeros(Nsubj+1)
    >>> for i in range(len(CVList)):
    >>>     cnt[CVList[i][1]] += 1
    >>> print(cnt[1:])
    >>> cnt = np.zeros(Nsubj+1)
    >>> for i in range(len(CVList)):
    >>>     cnt[CVList[i][2]] += 1
    >>> print(cnt[1:])
    
    '''
    # Check some input arguments
    if train_size<=0:
        raise ValueError("training set size percentage cannot be lower than 0%")
    elif train_size>=1:
        raise ValueError("training set size percentage should be in (0,1)")

    if val_size<=0:
        raise ValueError("validation set size percentage cannot be lower than 0%")
    elif val_size>=1:
        raise ValueError("validation set size percentage should be in (0,1)")

    if test_size<=0:
        raise ValueError("test set size percentage cannot be lower than 0%")
    elif test_size>=1:
        raise ValueError("test set size percentage should be in (0,1)")

    if (train_size + val_size + test_size) > 1:
        raise ValueError("train+val+test cannot exceed 1")
    
    # preallocate list of split.
    # Each split is represented by a set of three lists:
    #   the first for train,
    #   the second for validation,
    #   the third for test
    partition_list = [[[],[],[]] for i in range(folds)]
    
    # initialize list of subjects
    if isinstance(subj,(int, float)):
        Nsubj = int(subj)
        start_idx = start_with
        end_idx = start_idx + Nsubj
        subj_list = [i for i in range(start_idx, end_idx)]
    else:
        subj_list = subj
        Nsubj = len(subj_list)
    subj_list_set = set(subj_list)

    # Step 0 --> calculate some variables
    test_num = int(Nsubj*test_size)
    val_num = int(Nsubj*val_size)
    if (train_size + val_size + test_size)==1:
        train_num = Nsubj - test_num -val_num
    else:
        train_num = int(Nsubj*train_size)

    assign_tr = (folds*train_num)/Nsubj
    assign_va = (folds*val_num)/Nsubj
    assign_te = (folds*test_num)/Nsubj

    # Set seed if given
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # CASE 1: RANDOM SAMPLING --> homogeneous_sampling is False
    # if random sampling is choosen, we simply shuffle the subject list
    # fill the split with the first/last elements of the shuffled list,
    # and repeat the process.
    if not homogeneous_sampling:
        slice1 = train_num
        slice2 = slice1 + val_num
        slice3 = test_num
        for split in range(folds):
            random.shuffle(subj_list)
            partition_list[split][0] = subj_list[:slice1]
            partition_list[split][1] = subj_list[slice1:slice2]
            # using [-slice3:] makes test sets equals when changing the train/val size
            partition_list[split][2] = subj_list[-slice3:]
        return partition_list

    # CASE 2: HOMOGENEOUS RANDOM SAMPLING --> homogeneous_sampling is True
    # if homogeneous_sampling is choosen
    # we need to perform a much more complicated process
    
    # Step 1 --> shuffle list
    random.shuffle(subj_list)
    
    # Step 2 --> assign subjects to the test set
    split_list = np.arange(folds)
    available_splits = np.ones(folds, dtype=np.bool)
    counter = np.zeros(folds)
    assigned_idx = {i: [] for i in subj_list}
    
    # the idea is to iterate through the subject list and assign the subject
    # to a target number of random splits.
    for n, i in enumerate(subj_list):

        idx1 = np.array([], dtype=np.int64)
        idx2 = np.array([], dtype=np.int64)

        # define the number of split to extract
        curr_assign = round(assign_te*(n+1)) - round(assign_te*n)

        # if there are splits that are were chosen few times
        # (with a difference from the maximum number of filled elements >= 2)
        # select them to have a more uniform distribution
        mask = np.ones(folds, dtype=np.bool)
        divergent_idx = (counter - counter.max())<= -2
        if (divergent_idx.sum()>0):
            if divergent_idx.sum() < curr_assign:
                idx1 = np.nonzero(divergent_idx)[0]
                curr_assign -= len(idx1)
                mask[divergent_idx] = False

        # if the previous step included a number of splits lower than curr_assign
        # select the remaining splits randomly as usual
        if curr_assign>=0:
            mask[(counter - counter.min())>=2]= False
            mask = np.logical_and(mask, available_splits)
            possible_splits = split_list[mask]
            idx2 =  np.random.choice(possible_splits, curr_assign, replace=False)

        # assign the subject to the selected splits and keep track of the splits
        # where it is placed. This is important to avoid subject repetition between
        # the train/validation/test set of the same split
        idx = np.concatenate((idx1,idx2))
        assigned_idx[i] = np.sort(idx)
        for split in idx:
            partition_list[split][2].append(i)
        counter[idx] += 1
        # if a split is filled, exclude it from the next assignments
        available_splits[counter>=test_num] = False

    # Step 3 --> shuffle list again
    random.shuffle(subj_list)

    # Step 4 --> assign subjects to the validation set
    split_list = np.arange(folds)
    available_splits = np.ones(folds, dtype=np.bool)
    counter = np.zeros(folds)

    # This process is the same as before, with the only difference that we need
    # to exclude the splits where the same subjects is assigned in the test set
    for n, i in enumerate(subj_list):

        idx1 = np.array([], dtype=np.int64)
        idx2 = np.array([], dtype=np.int64)
        
        curr_assign = round(assign_va*(n+1)) - round(assign_va*n)
        mask = np.ones(folds, dtype=np.bool)
        mask[assigned_idx[i]] = False
        mask = np.logical_and(mask, available_splits)

        divergent_idx = (counter - counter.max())<= -2
        if (divergent_idx.sum()>0):
            if divergent_idx.sum() < curr_assign:
                idx1 = np.nonzero(np.logical_and(mask,divergent_idx))[0]
                curr_assign -= len(idx1)
                mask[divergent_idx] = False

        if curr_assign>=0:
            possible_splits = split_list[mask]
            idx2 =  np.random.choice(
                possible_splits,
                min(len(possible_splits), curr_assign),
                replace=False
            )
        
        idx = np.concatenate((idx1,idx2))
        assigned_idx[i] = np.concatenate((assigned_idx[i], idx))
        assigned_idx[i] = np.sort(assigned_idx[i])
        for split in idx:
            partition_list[split][1].append(i)
        counter[idx] += 1
        available_splits[counter>=val_num] = False

    
    # if the length of the validation set is not the correct
    # for all CV splits, fill them by taking random subjects
    # from the entire list, excluding only those
    # assigned to the test set of that partition
    for split in range(folds):
        if len(partition_list[split][1]) != val_num:
            target_num = val_num - len(partition_list[split][1])
            target_list = list(
                subj_list_set.difference(
                    set(partition_list[split][1] + partition_list[split][2])
                )
            )
            correct_subj = np.random.choice(target_list, target_num, replace=False)
            for i in correct_subj:
                assigned_idx[i] = np.concatenate((assigned_idx[i], np.array([split])))
                assigned_idx[i] = np.sort(assigned_idx[i])
            partition_list[split][1] += correct_subj.tolist()

    # Step 6 --> assign subjects to the training set
    # if the sum of all sizes is one, we just assign the remaining subjects to the
    # training set, otherwise we repeat the same thing done for the validation set
    if (train_size + val_size + test_size)==1:
        for split in range(folds):
            partition_list[split][0] = list(
                subj_list_set.difference(
                    set(partition_list[split][1] + partition_list[split][2])
                )
            )
    else:
        random.shuffle(subj_list)
        split_list = np.arange(folds)
        available_splits = np.ones(folds, dtype=np.bool)
        counter = np.zeros(folds)
        for n, i in enumerate(subj_list):

            idx1 = np.array([], dtype=np.int64)
            idx2 = np.array([], dtype=np.int64)
            
            curr_assign = round(assign_tr*(n+1)) - round(assign_tr*n)
            mask = np.ones(folds, dtype=np.bool)
            mask[assigned_idx[i]] = False
            mask = np.logical_and(mask, available_splits)

            divergent_idx = (counter - counter.max())<= -2
            if (divergent_idx.sum()>0):
                if divergent_idx.sum() < curr_assign:
                    idx1 = np.nonzero(np.logical_and(mask,divergent_idx))[0]
                    curr_assign -= len(idx1)
                    mask[divergent_idx] = False
            
            if curr_assign>=0:
                possible_splits = split_list[mask]
                idx2 =  np.random.choice(
                    possible_splits,
                    min(len(possible_splits), curr_assign),
                    replace=False
                )
            
            idx = np.concatenate((idx1,idx2))
            for split in idx:
                partition_list[split][0].append(i)
            counter[idx] += 1
            available_splits[counter>=train_num] = False

        # apply post correction as in the validation set
        for split in range(folds):
            if len(partition_list[split][0]) != train_num:
                target_num = train_num - len(partition_list[split][0])
                target_list = list(
                    subj_list_set.difference(
                        set(partition_list[split][0] +\
                            partition_list[split][1] +\
                            partition_list[split][2])
                    )
                )
                correct_subj = np.random.choice(target_list, target_num, replace=False)
                partition_list[split][0] += correct_subj.tolist()
        
    return partition_list


def merge_partition_lists_noshift(
    l1: list[list[int], list[int], list[int]], 
    l2: list[list[int], list[int], list[int]]
) -> list[list[int], list[int], list[int]]:
    '''
    merge_partition_lists_noshift merges the rows of two partition list with the
    same length. Partitions are supposed to come from the 
    create_nested_kfold_subject_split function.

    Parameters
    ----------
    l1: list
        The first partition list. It must be a list of three integer lists,
        the first with the subject's training IDs, the second with the subject's
        validation IDs, and the third with the subject's test IDs. For example,
        [ [[1,2,3], [4,5] ,[6,7]], [[4,5],[1,2,3],[6,7]] ]
    l2: list
        The second partition list. Same as l1. The length of l1 must be equal 
        to the length of l2

    Returns
    -------
    new_list: list
        The merged partition list. The structure is the same as l1 and l2.
    
    '''
    if len(l1) != len(l2):
        ValueError('the two partition lists must be of the same length')

    new_list = [[[None],[None],[None]] for i in range(len(l1))]
    for i in range(len(l1)):
        new_list[i][0] = list(set(l1[i][0] + l2[i][0]))
        new_list[i][1] = list(set(l1[i][1] + l2[i][1]))
        new_list[i][2] = list(set(l1[i][2] + l2[i][2]))

    return new_list


def merge_partition_lists(
    l1: list[list[int], list[int], list[int]], 
    l2: list[list[int], list[int], list[int]],
    folds: int,
    folds_nested: int
) -> list[list[int], list[int], list[int]]:
    '''
    merge_partition_lists merges the rows of two partition list with the
    same length. Partitions are supposed to come from the 
    create_nested_kfold_subject_split function.
    To avoid the creation of sets with too different size
    l2 rows will be selected by shifting 1 folds + 1 nested folds. 
    For example, in case of a 10 fold CV with 5 nested folds, the following 
    couples of rows will be selected:
    
        [0 6],  [1 7],  [2 8],  [3 9],  [4 5],
        [5 11], [6 12], [7 13], [8 14], [9 10],
        [10 16],[11 17],[12 18],[13 19],[14 15],
        [15 21],[16 22],[17 23],[18 24],[19 20],
        [20 26],[21 27],[22 28],[23 29],[24 25],
        [25 31],[26 32],[27 33],[28 34],[29 30],
        [30 36],[31 37],[32 38],[33 39],[34 35],
        [35 41],[36 42],[37 43],[38 44],[39 40],
        [40 46],[41 47],[42 48],[43 49],[44 45],
        [45 1], [46 2], [47 3], [48 4], [49 0],

    Parameters
    ----------
    l1: list
        The first partition list. It must be a list of three integer lists,
        the first with the subject's training IDs, the second with the subject's
        validation IDs, and the third with the subject's test IDs. For example,
        [ [[1,2,3], [4,5] ,[6,7]], [[4,5],[1,2,3],[6,7]] ]
    l2: list
        The second partition list. Same as l1. The length of l1 must be equal 
        to the length of l2
    folds: int
        The number of outer folds. It must be the same parameter given to the 
        create_nested_kfold_subject_split function.
    folds_nested: int
        The number of nested folds. It must be the same parameter given to the 
        create_nested_kfold_subject_split function.

    Returns
    -------
    new_list: list
        The merged partition list with fusion shift paradigm. 
        The structure is the same as l1 and l2.
        
    '''
    if len(l1) != len(l2):
        ValueError('the two partition lists must be of the same length')

    new_list = [[[None],[None],[None]] for i in range(len(l1))]
    for i in range(folds):
        for k in range(folds_nested):
            l1_idx = folds_nested*i+k
            if k == folds_nested-1:
                l2_idx = folds_nested*(i+1)
            else: 
                l2_idx = folds_nested*(i+1) + k + 1
            if i == folds - 1:
                l2_idx = 0 if k == folds_nested-1 else k + 1
            new_list[l1_idx][0] = list(set(l1[l1_idx][0] + l2[l2_idx][0]))
            new_list[l1_idx][1] = list(set(l1[l1_idx][1] + l2[l2_idx][1]))
            new_list[l1_idx][2] = list(set(l1[l1_idx][2] + l2[l2_idx][2]))

    return new_list
