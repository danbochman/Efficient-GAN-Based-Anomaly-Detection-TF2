from absl import app
from absl import flags

from bigan.bigan_trainer import train
from dataloader.image_generators import train_val_test_image_generator

FLAGS = flags.FLAGS
flags.DEFINE_integer("training_steps", 100000, "Number of training steps")
flags.DEFINE_integer("crop_size", 128, "Shape of (S, S) to take from image")
flags.DEFINE_integer("latent_dim", 128, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 32, "Size of training batches")
flags.DEFINE_float("lr", 0.0002, "Learning rate for optimizers")
flags.DEFINE_integer("display_step", 100, "Writing frequency for TensorBoard")
flags.DEFINE_float("resize", 0.5, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")
flags.DEFINE_integer("save_checkpoint_every_n_steps", 10000, "Frequency for saving model checkpoints")
flags.DEFINE_string("logs_dir", './bigan/logs', "relative dir path to save TB events and checkpoints")
flags.DEFINE_string("data_path", None, "absolute dir path for image dataset")


def main(argv=None):
    # init data generator (64x64 or 32x32 size images are recommended - modify with crop_size & resize)
    train_img_gen, test_img_gen = train_val_test_image_generator(data_path=FLAGS.data_path,
                                                                 crop_size=FLAGS.crop_size,
                                                                 batch_size=FLAGS.batch_size,
                                                                 resize=FLAGS.resize,
                                                                 normalize=True,
                                                                 val_frac=0.0)

    # infer input shape from crop size and resizing factor
    s = int(FLAGS.crop_size * FLAGS.resize)
    input_shape = (s, s, 1)

    # train on data generator
    train(data_generator=train_img_gen,
          input_shape=input_shape,
          training_steps=FLAGS.training_steps,
          latent_dim=FLAGS.latent_dim,
          lr=FLAGS.lr,
          display_step=FLAGS.display_step,
          save_checkpoint_every_n_steps=FLAGS.save_checkpoint_every_n_steps,
          logs_dir=FLAGS.logs_dir)


if __name__ == '__main__':
    app.run(main)
