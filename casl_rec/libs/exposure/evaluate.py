import numpy as np
import bottleneck as bn
import tensorflow as tf


def binary_crossentropy(y_true, y_pred):
    '''
        The tensorflow style binary crossentropy
    '''
    loss = -tf.reduce_mean(
        tf.reduce_sum(
            y_true * tf.log(tf.maximum(y_pred, 1e-10)) + (1-y_true) * 
            tf.log(tf.maximum(1-y_pred, 1e-10)), axis=-1
        ))
    return loss


def multinomial_crossentropy(y_true, y_pred):
    loss = -tf.reduce_mean(tf.reduce_sum(
            y_true * tf.log(tf.maximum(y_pred, 1e-10)), axis=1
    ))
    return loss


def mse(y_true, y_pred):
    '''
        The tensorflow style mean squareed error
    '''
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
    return loss


def weighted_mse_generator(weight_0, weight_1):
    '''
        Should return in a partial manner.
    '''
    def weight_mse(y_true, y_pred):
        shared = tf.square(y_true - y_pred)
        loss_0 = tf.multiply((weight_0*(1 - y_true)), shared)
        loss_1 = tf.multiply((weight_1*y_true), shared)
        loss = tf.reduce_mean(tf.reduce_sum(loss_0 + loss_1, axis=-1))
        return loss
    return weight_mse


def Recall_at_k(y_true, y_pred, k):
    '''
        Recall evaluated at top K (Implicit).
    '''
    batch_size = y_pred.shape[0]
    topk_idxes = bn.argpartition(-y_pred, k, axis=1)[:, :k]
    y_pred_bin = np.zeros_like(y_pred, dtype=np.bool)
    y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
    y_true_bin = (y_true > 0)
    hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
    recall = np.mean(hits/np.minimum(k, np.sum(y_true_bin, axis=1)))
    return recall


def NDCG_at_k(y_true, y_pred, k):
    '''
        NDCG evaluated at top K (Implicit).
    '''
    batch_size = y_pred.shape[0]
    topk_idxes_unsort = bn.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    y_true_topk = y_true[np.arange(batch_size)[:, None], topk_idxes]
    y_true_bin = (y_true > 0).astype(np.float32)
    weights = 1./np.log2(np.arange(2, k + 2))
    DCG = np.sum(y_true_topk*weights, axis=-1)
    normalizer = np.array([np.sum(weights[:int(n)]) for n in np.minimum(k, np.sum(y_true_bin, axis=-1))])
    NDCG = np.mean(DCG/normalizer)
    return NDCG


def DCG_at_k(y_true, y_pred, k):
    '''
        DCG evaluated at top K.
    '''
    batch_size = y_pred.shape[0]
    topk_idxes_unsort = bn.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    topk_y = y_true[np.arange(batch_size)[:, None], topk_idxes]
    weights = 1./np.log2(np.arange(2, k + 2)) 
    return np.sum(topk_y*weights, axis=-1)


def NDCG_at_k_explicit(y_true, y_pred, k):
    '''
        NDCG evaluated at top K (Explicit).
        For binary data, use NDCG_at_k for efficiency.
    '''
    DCG = DCG_at_k(y_true, y_pred, k)
    normalizer = DCG_at_k(y_true, y_true, k)
    return np.mean(DCG/normalizer)


def Recall_at_k_explicit(y_true, y_pred, k, thres=4.5):
    '''
        Recall evaluated at top K (Implicit).
    '''
    y_true = (y_true > thres).astype(np.int32)
    rem_users = (np.sum(y_true, axis=-1) > 0)
    y_true = y_true[rem_users];  y_pred = y_pred[rem_users]
    return Recall_at_k(y_true, y_pred, k)


def EvaluateModel(eval_model, eval_gen, eval_func, k):
    '''
        Evaluate the trained model.
    '''
    metric_list = []
    for (obs_records, unk_true) in eval_gen:
        unk_pred = eval_model.predict_on_batch(obs_records)
        unk_pred[obs_records.astype(np.bool)] = -np.inf
        metric_list.append(eval_func(unk_true, unk_pred, k))
    metric = np.mean(metric_list)
    return metric