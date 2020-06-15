from resnet import resnet_v2_50, resnet_arg_scope
import tensorflow as tf
import tensorflow.contrib.slim as slim
from ops import conv, swish, bifpn_layer, batchnorm
from config import *


def class_subnet(inputs, is_training):
    with tf.variable_scope("class_subnet", reuse=tf.AUTO_REUSE):
        for i in range(D_class):
            inputs = swish(batchnorm("bn1"+str(i), conv("conv1"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
            inputs = swish(batchnorm("bn2"+str(i), conv("conv2"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
            inputs = swish(batchnorm("bn3"+str(i), conv("conv3"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
            inputs = swish(batchnorm("bn4"+str(i), conv("conv4"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
        inputs = conv("conv5", inputs, K*A, 3, 1, "SAME", True)
        H, W = tf.shape(inputs)[1], tf.shape(inputs)[2]
        inputs = tf.reshape(inputs, [-1, H * W * A, K])
    return inputs

def box_subnet(inputs, is_training):
    with tf.variable_scope("box_subnet", reuse=tf.AUTO_REUSE):
        for i in range(D_class):
            inputs = swish(batchnorm("bn1"+str(i), conv("conv1"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
            inputs = swish(batchnorm("bn2"+str(i), conv("conv2"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
            inputs = swish(batchnorm("bn3"+str(i), conv("conv3"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
            inputs = swish(batchnorm("bn4"+str(i), conv("conv4"+str(i), inputs, W_bifpn, 3, 1, "SAME"), is_training))
        inputs = conv("conv5", inputs, 4*A, 3, 1, "SAME")
        H, W = tf.shape(inputs)[1], tf.shape(inputs)[2]
        inputs = tf.reshape(inputs, [-1, H * W * A, 4])
    return inputs

def backbone(inputs, is_training):
    arg_scope = resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        _, end_points = resnet_v2_50(inputs, is_training=is_training)
    C3 = end_points["resnet_v2_50/block2/unit_3/bottleneck_v2"]
    C4 = end_points["resnet_v2_50/block3/unit_5/bottleneck_v2"]
    C5 = end_points["resnet_v2_50/block4/unit_3/bottleneck_v2"]
    P3 = swish(batchnorm("bn1", conv("conv3", C3, W_bifpn, 3, 1, "SAME"), is_training))
    P4 = swish(batchnorm("bn2", conv("conv4", C4, W_bifpn, 3, 1, "SAME"), is_training))
    P5 = swish(batchnorm("bn3", conv("conv5", C5, W_bifpn, 3, 1, "SAME"), is_training))
    P6 = swish(batchnorm("bn4", conv("conv6", C5, W_bifpn, 3, 2, "SAME"), is_training))
    P7 = swish(batchnorm("bn5", conv("conv7", P6, W_bifpn, 3, 2, "SAME"), is_training))
    for i in range(D_bifpn):
        P3, P4, P5, P6, P7 = bifpn_layer("bifpn"+str(i), P3, P4, P5, P6, P7, is_training)

    P3_class_logits = class_subnet(P3, is_training)
    P3_box_logits = box_subnet(P3, is_training)

    P4_class_logits = class_subnet(P4, is_training)
    P4_box_logits = box_subnet(P4, is_training)

    P5_class_logits = class_subnet(P5, is_training)
    P5_box_logits = box_subnet(P5, is_training)

    P6_class_logits = class_subnet(P6, is_training)
    P6_box_logits = box_subnet(P6, is_training)

    P7_class_logits = class_subnet(P7, is_training)
    P7_box_logits = box_subnet(P7, is_training)
    class_logits = tf.concat([P3_class_logits, P4_class_logits, P5_class_logits, P6_class_logits, P7_class_logits], axis=1)
    box_logits = tf.concat([P3_box_logits, P4_box_logits, P5_box_logits, P6_box_logits, P7_box_logits], axis=1)
    class_logits_dict = {"P3": P3_class_logits, "P4": P4_class_logits, "P5": P5_class_logits,
                         "P6": P6_class_logits, "P7": P7_class_logits}
    box_logits_dict = {"P3": P3_box_logits, "P4": P4_box_logits, "P5": P5_box_logits,
                       "P6": P6_box_logits, "P7": P7_box_logits}
    return class_logits, box_logits, class_logits_dict, box_logits_dict

# inputs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
# is_training = tf.placeholder(tf.bool)
# backbone(inputs, is_training)