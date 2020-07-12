import os

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from bigan.bigan_model import BiGAN

PHYSICAL_DEVICES = tf.config.experimental.list_physical_devices('GPU')
if len(PHYSICAL_DEVICES) > 0:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)


def train(data_generator,
          input_shape,
          training_steps,
          latent_dim,
          lr,
          display_step,
          save_checkpoint_every_n_steps,
          logs_dir):
    """
    Trainer function for the BiGAN model, all function parameters explanations are detailed in the egbad_train_main.py flags.
    The function flow is as follows:
    1. Initializes a data generator from the data_path (while preprocessing the images),
    2. initializes the optimizers, tensorboard & checkpoints writers
    3. restores from checkpoint if exists
    4. Training loop:
        - grab img batch from generator
        - encode img with E(x)
        - sample from latent space (create z)
        - generate img from z with generator D(z)
        - reconstruct original image from encoding with generator (for display)
        - feed (image, latent representation) through discriminator
        - compute losses
        - take optimizer steps for each submodel
        - write to tensorboard / save checkpoint every n steps
    """

    # init BiGAN model
    egbad = BiGAN(input_shape=input_shape, latent_dim=latent_dim)
    gen = egbad.Gz
    enc = egbad.Ex
    dis = egbad.Dxz

    # optimizers
    optimizer_dis = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5,
                                             name='dis_optimizer')  # SGD is also recommended for dis
    optimizer_gen = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, name='gen_optimizer')
    optimizer_enc = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, name='enc_optimizer')

    # summary writers
    scalar_writer = tf.summary.create_file_writer(logs_dir + '/scalars')
    image_writer = tf.summary.create_file_writer(logs_dir + '/images')

    # checkpoint writer
    checkpoint_dir = logs_dir + '/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer_gen,
                                     discriminator_optimizer=optimizer_dis,
                                     encoder_optimizer=optimizer_enc,
                                     generator=gen,
                                     discriminator=dis,
                                     encoder=enc)

    # restore from checkpoint if exists
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    progres_bar = Progbar(training_steps)
    for step in range(training_steps):
        progres_bar.update(step)
        img_batch, label_batch = next(data_generator)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape, tf.GradientTape() as enc_tape:
            # encoder
            z_gen = enc(img_batch)

            # generator
            z = tf.random.normal((img_batch.shape[0], latent_dim))
            x_gen = gen(z)
            x_rec = gen(z_gen)

            # discriminator
            logits_real, features_real = dis([img_batch, z_gen])
            logits_fake, features_fake = dis([x_gen, z])

            # losses (label smoothing may be helpful)
            # discriminator
            loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                                  logits=logits_real))
            loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                                  logits=logits_fake))
            loss_dis = loss_dis_gen + loss_dis_enc

            # generator
            loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),
                                                                              logits=logits_fake))
            # encoder
            loss_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_real),
                                                                              logits=logits_real))

        # compute gradients
        grad_gen = gen_tape.gradient(loss_gen, gen.trainable_variables)
        grad_dis = dis_tape.gradient(loss_dis, dis.trainable_variables)
        grad_enc = enc_tape.gradient(loss_enc, enc.trainable_variables)

        # apply gradients
        optimizer_gen.apply_gradients(zip(grad_gen, gen.trainable_variables))
        optimizer_dis.apply_gradients(zip(grad_dis, dis.trainable_variables))
        optimizer_enc.apply_gradients(zip(grad_enc, enc.trainable_variables))

        # write summaries
        if step % display_step == 0:
            with scalar_writer.as_default():
                # discriminator parts
                tf.summary.scalar("loss_discriminator", loss_dis, step=step)
                tf.summary.scalar("loss_dis_enc", loss_dis_enc, step=step)
                tf.summary.scalar("loss_dis_gen", loss_dis_gen, step=step)
                # generator
                tf.summary.scalar("loss_generator", loss_gen, step=step)
                # encoder
                tf.summary.scalar("loss_encoder", loss_enc, step=step)

            with image_writer.as_default():
                # [-1, 1] -> [0, 255]
                orig_display = tf.cast((img_batch + 1) * 127.5, tf.uint8)
                gen_display = tf.cast((x_gen + 1) * 127.5, tf.uint8)
                rec_display = tf.cast((x_rec + 1) * 127.5, tf.uint8)
                tf.summary.image('Original', orig_display, step=step, max_outputs=3)
                tf.summary.image('Generated', gen_display, step=step, max_outputs=3)
                tf.summary.image('Reconstructed', rec_display, step=step, max_outputs=3)

        if step % save_checkpoint_every_n_steps == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
