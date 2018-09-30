from model import Pix2Pix
from data_utils import get_train_test_files, get_data_gen
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# params

batch_size = 2
timesteps = 2
im_width = im_height = 256
# end params


train_files, test_files = get_train_test_files()
train_gen = get_data_gen(files=train_files, timesteps=timesteps, batch_size=batch_size, im_size=(im_width, im_height))
gan = Pix2Pix(im_height=im_height, im_width=im_width, lookback=timesteps-1)
print("Generator Summary")
gan.generator.summary()
print()
print("Discriminator Summary")
gan.discriminator.summary()
print()
print("Combined Summary")
gan.combined.summary()
gan.train(train_gen, epochs=600, batch_size=batch_size, save_interval=200, save_file_name="r_p2p_gen_t2.model")