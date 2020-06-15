import tensorflow as tf
import numpy as np
from config import W_bifpn, EPSILON

def batchnorm(scope_bn, x, train_phase):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv(name, inputs, nums_out, k_size, stride, padding, is_final=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.01))
        if is_final:
            b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([-np.log((1 - 0.01) / 0.01)]))
        else:
            b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, stride, stride, 1], padding)
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def swish(inputs):
    return tf.nn.swish(inputs)

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def resize(inputs, factor=2.):
    H, W = int(inputs.shape[1]), int(inputs.shape[2])
    return tf.image.resize_nearest_neighbor(inputs, [int(H*factor), int(W*factor)])

def bifpn_layer(name, p3, p4, p5, p6, p7, train_phase):
    with tf.variable_scope(name):
        with tf.variable_scope("intermediate_p6"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p6 + w2 * resize(p7)
            P_6_td = swish(batchnorm("bn1", conv("conv6_td", temp/(w1 + w2 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("intermediate_p5"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p5 + w2 * resize(P_6_td)
            P_5_td = swish(batchnorm("bn1", conv("conv5_td", temp / (w1 + w2 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("intermediate_p4"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p4 + w2 * resize(P_5_td)
            P_4_td = swish(batchnorm("bn1", conv("conv4_td", temp / (w1 + w2 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("output_p3"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p3 + w2 * resize(P_4_td)
            P_3_out = swish(batchnorm("bn1", conv("conv3_out", temp / (w1 + w2 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("output_p4"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w3 = tf.get_variable("W3", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p4 + w2 * P_4_td + w3 * resize(P_3_out, factor=0.5)
            P_4_out = swish(batchnorm("bn1", conv("conv4_out", temp / (w1 + w2 + w3 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("output_p5"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w3 = tf.get_variable("W3", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p5 + w2 * P_5_td + w3 * resize(P_4_out, factor=0.5)
            P_5_out = swish(batchnorm("bn1", conv("conv5_out", temp / (w1 + w2 + w3 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("output_p6"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w3 = tf.get_variable("W3", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p6 + w2 * P_6_td + w3 * resize(P_5_out, factor=0.5)
            P_6_out = swish(batchnorm("bn1", conv("conv6_out", temp / (w1 + w2 + w3 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
        with tf.variable_scope("output_p7"):
            w1 = tf.get_variable("W1", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            w2 = tf.get_variable("W2", [1, 1, 1, 1], initializer=tf.initializers.he_normal())
            temp = w1 * p7 + w2 * resize(P_6_out, factor=0.5)
            P_7_out = swish(batchnorm("bn1", conv("conv7_out", temp / (w1 + w2 + EPSILON), W_bifpn, k_size=3, stride=1, padding="SAME"), train_phase))
    return P_3_out, P_4_out, P_5_out, P_6_out, P_7_out

def smooth_l1(inputs):
    loss = tf.where(tf.less(tf.abs(inputs), 1.0), 0.5 * tf.square(inputs), tf.abs(inputs) - 0.5)
    loss = tf.reduce_sum(loss, axis=2)
    return loss


def focal_loss(logits, labels, alpha=0.25, gamma=1.5):
    pos_pt = tf.clip_by_value(tf.nn.sigmoid(logits), 1e-10, 0.999)
    fl = labels * tf.log(pos_pt) * tf.pow(1 - pos_pt, gamma) * alpha + (1 - labels) * tf.log(1 - pos_pt) * tf.pow(pos_pt, gamma) * (1 - alpha)
    fl = -tf.reduce_sum(fl, axis=2)
    return fl

def offset2bbox(anchors, t_bbox):
    bbox_x = t_bbox[:, 0:1] * anchors[:, 2:3] + anchors[:, 0:1]
    bbox_y = t_bbox[:, 1:2] * anchors[:, 3:4] + anchors[:, 1:2]
    bbox_w = tf.exp(t_bbox[:, 2:3]) * anchors[:, 2:3]
    bbox_h = tf.exp(t_bbox[:, 3:4]) * anchors[:, 3:4]
    x1, y1 = bbox_x - bbox_w / 2, bbox_y - bbox_h / 2
    x2, y2 = bbox_x + bbox_w / 2, bbox_y + bbox_h / 2
    return tf.concat((x1, y1, x2, y2), axis=1)

def top_k_score_bbox(pred_score, pred_bbox, anchors, threshold=0.05, k=1000):
    pred_score_obj = tf.reduce_max(pred_score, axis=1)
    idx = tf.where(tf.greater(pred_score_obj, threshold))[:, 0]
    threshold_score = tf.nn.embedding_lookup(pred_score_obj, idx)
    threshold_bbox = tf.nn.embedding_lookup(pred_bbox, idx)
    threshold_anchors = tf.nn.embedding_lookup(anchors, idx)
    threshold_nums = tf.shape(threshold_score)[0]
    k = tf.where(tf.greater(threshold_nums, k), k, threshold_nums)
    topK_score, topK_indx = tf.nn.top_k(threshold_score, k)
    topK_bbox = tf.nn.embedding_lookup(threshold_bbox, topK_indx)
    topK_anchors = tf.nn.embedding_lookup(threshold_anchors, topK_indx)
    pred_score_idx = tf.nn.embedding_lookup(idx, topK_indx)
    topK_class_score = tf.nn.embedding_lookup(pred_score, pred_score_idx)
    topK_class_labels = tf.argmax(topK_class_score, axis=1)
    return topK_score, topK_bbox, topK_anchors, topK_class_labels


