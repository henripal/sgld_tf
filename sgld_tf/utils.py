import os
import numpy as np
import sgld_tf
from tensorflow.python import pywrap_tensorflow
from multiprocessing import Pool

import statsmodels.tsa.stattools as st

def ordered_chkpoints_from_dir(directory):
	ckpt_set = {'.'.join(fn.split('.')[:2]) for fn in os.listdir(directory) if fn[:3] == 'mod'}
	return sorted(ckpt_set, key=lambda x: int(x.split('-')[1]))

def aggregate_predictions(root, names, train_data, train_labels, eval_data,
							eval_labels, method):
    cnn = sgld_tf.CNN(train_data, train_labels, eval_data, eval_labels,
                      bs=512, model_dir=root, optimize_method=method,
                      learning_rate=0.001, save_checkpoint_steps=10,
                      keep_checkpoint_max=100, shuffle=True)
    all_model_probabilities = []
    all_model_predictions = []

    for name in names:
        single_model_probabilities = []
        predictions = cnn.estimator.predict(input_fn = cnn.eval_input_fn,
                                            checkpoint_path=root + name,
                                            yield_single_examples=False)
        n = eval_labels.shape[0]
        label_batches = (eval_labels[i:i+cnn.bs] for i in np.arange(0, n, cnn.bs))
        for labels, prediction in zip(label_batches, predictions):
            single_model_probabilities.append(prediction['probabilities'][:len(labels)])

        all_model_probabilities.append(np.vstack(single_model_probabilities))
        all_model_predictions.append(np.argmax(np.vstack(single_model_probabilities), axis=1))

    summed_probabilities = np.sum(np.array(all_model_probabilities), axis=0)
    predicted_labels = np.argmax(summed_probabilities, axis=1)
    accuracy = np.mean(predicted_labels == eval_labels)

    return accuracy, np.array(all_model_predictions)

def mess(multi_model_predictions, batch_size):
    """
    inputs:
        multi_model_predictions: a list of n p-dimensional predictions
        batch_size: the batch size to compute mESS
    """
    multi_model_predictions = np.array(multi_model_predictions)

    n, p = multi_model_predictions.shape

    var = np.var(multi_model_predictions, axis=0)

    batches = np.array_split(multi_model_predictions, n//batch_size, axis=0)
    batch_means = [np.mean(batch, axis=0) for batch in batches]
    batch_vars = batch_size * np.var(np.array(batch_means), axis=0)

    log_var = np.sum(np.log(var)) / p
    log_bvar = np.sum(np.log(batch_vars)) / p

    return n, n * np.exp(log_var - log_bvar)


def analyze_directory(directory, stdev_n_from_last=100, delta=20):
    modelnames = {'.'.join(fname.split('.')[:2]) for fname in os.listdir(directory) if fname[:5] =='model'}
    mn = list(sorted(modelnames, key=lambda x:int(x.split('-')[-1])))
    histo = []
    model_file = directory + mn[0]
    reader = pywrap_tensorflow.NewCheckpointReader(model_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    valid_keys = []
    for key in var_to_shape_map:
        if (key[:3] == 'con' or key[:3] == 'den') and len(key.split('/')) == 2:
            valid_keys.append(key)
    print(valid_keys)

    for chkpt in mn:
        model_file = directory + chkpt
        reader = pywrap_tensorflow.NewCheckpointReader(model_file)
        model_vector = []
        for key in valid_keys:
            model_vector.append(reader.get_tensor(key).ravel())

        out = np.hstack(model_vector)
        histo.append(out)
    histo = np.vstack(histo)
    print('hist shape: ', histo.shape)
    stdevs = np.std(histo[-stdev_n_from_last:], axis=0)
    print('max std: ', np.max(stdevs))

    return histo, stdevs

def mess_correl(data):
    n, p = data.shape
    data = data.T
    pool = Pool(16)
    ess = pool.map(univess, [row for row in data])

    ess = np.array(ess)
    isf = np.isfinite(ess)
    print("percentage of finite: ", np.mean(isf))
    ess = ess[np.isfinite(ess)]

    return np.min(ess), np.exp(np.mean(np.log(ess)))

def univess(ts):
    acf = st.acf(ts, nlags=20)
    ind = np.argmax(acf < 0)
    if ind == 0: ind = len(acf)
    sum_acf = np.sum(acf[1:ind])
    return len(ts) / (1+2*sum_acf)

def split_mess(data, n_splits, n_size):
    results = []
    for i in range(n_splits):
        print("split: ", i)
        split_data = data[i*n_size:(i+1)*n_size]
        result = mess_correl(split_data)
        print(result)
        results.append(result)

    return results
