from absl import app
from absl import flags

from bigan.egbad_eval import eval
from dataloader.image_generators import train_val_test_image_generator
from eval_utils.visualization_utils import save_precision_recall_curve

FLAGS = flags.FLAGS
flags.DEFINE_string("method", "fm", "choose discriminator score method fm/ce")
flags.DEFINE_float("weight", 0.1, "weight for discriminator score vs reconstruction error")
flags.DEFINE_integer("crop_size", 128, "Shape of (S, S) to take from image")
flags.DEFINE_integer("latent_dim", 128, "Size of latent representation of model")
flags.DEFINE_integer("batch_size", 32, "Size of training batches")
flags.DEFINE_float("resize", 0.5, "Resizing factor for crops if necessary to fit in e.g. 64x64xc crops")
flags.DEFINE_string("logs_dir", './bigan/logs', "relative dir path which contains the checkpoint dir")
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

    # eval on data generator
    anomaly_scores, labels = eval(data_generator=test_img_gen,
                                  input_shape=input_shape,
                                  latent_dim=FLAGS.latent_dim,
                                  method=FLAGS.method,
                                  weight=FLAGS.weight,
                                  logs_dir=FLAGS.logs_dir)

    # plot precision recall curve
    save_precision_recall_curve(anomaly_scores, labels)


if __name__ == '__main__':
    app.run(main)
