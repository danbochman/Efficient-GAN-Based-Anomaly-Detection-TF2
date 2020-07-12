import os

import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tqdm import tqdm

from bigan.bigan_model import BiGAN

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def eval(data_generator,
         input_shape,
         latent_dim,
         method,
         weight,
         logs_dir):
    egbad = BiGAN(input_shape=input_shape, latent_dim=latent_dim)

    gen = egbad.Gz
    enc = egbad.Ex
    dis = egbad.Dxz

    # checkpoint writer
    checkpoint_dir = logs_dir + 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=gen,
                                     discriminator=dis,
                                     encoder=enc)

    # restore from checkpoint if exists
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    anomaly_scores = []
    labels = []
    for img_batch, label_batch in tqdm(data_generator):

        # generator reconstruction loss (can be L1 or L2)
        z_enc = enc(img_batch, training=False)
        x_rec = gen(z_enc, training=False)
        diff = img_batch - x_rec
        diff = Flatten()(diff)
        gen_score = tf.norm(diff, ord=1, axis=1, keepdims=False)

        # discriminator loss
        x_logits, x_features = dis([img_batch, z_enc], training=False)
        rec_logits, rec_features = dis([x_rec, z_enc], training=False)

        if method == 'ce':
            dis_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(rec_logits), logits=rec_logits)

        elif method == "fm":
            fm = x_features - rec_features
            fm = Flatten()(fm)
            dis_score = tf.norm(fm, ord=1, axis=1, keepdims=False)

        anomaly_score = (1 - weight) * gen_score + weight * dis_score
        labels.extend(label_batch)
        anomaly_scores.extend(anomaly_score)

    return anomaly_scores, labels
