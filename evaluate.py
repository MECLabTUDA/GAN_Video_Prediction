import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from moviepy.editor import CompositeVideoClip, ImageSequenceClip
from data_utils import get_data_gen, get_train_test_files, denormalize, VIDEO_KNOT, VIDEO_NEEDLE_PASSING, VIDEO_SUTURING
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras




# params

batch_size = 1
timesteps = 5
im_width = im_height = 256
# end params

def generate_video(saved_model_path, video_category=None):
    """Uses the trained model to predict the frames and produce a video out of them"""
    # load model
    model = load_model(saved_model_path)

    which_one = video_category
    train_files, test_files = get_train_test_files(which=which_one)
    test_gen = get_data_gen(files=test_files, timesteps=timesteps, batch_size=batch_size, im_size=(im_width, im_height))

    y_true = []
    y_pred = []

    for _ in range(200):
        x, y = next(test_gen)
        y_true.extend(y)

        predictions = model.predict_on_batch(x)
        y_pred.extend(predictions)


    clip1 = ImageSequenceClip([denormalize(i) for i in y_true], fps=5)
    clip2 = ImageSequenceClip([denormalize(i)for i in y_pred], fps=5)
    clip2 = clip2.set_position((clip1.w, 0))
    video = CompositeVideoClip((clip1, clip2), size=(clip1.w * 2, clip1.h))
    video.write_videofile("{}.mp4".format(which_one if which_one else "render"), fps=5)


def plot_different_models(timesteps = [5, 10]):
    """
    Compares ssim/psnr of different models. The models for each of the supplied timestap
    must be present
    param timesteps A list of numbers indicating the timesteps that were used for training different models
    """
    from skimage.measure import compare_psnr, compare_ssim
    psnrs = {}
    ssims = {}
    for ts in timesteps:
        model_name = "r_p2p_gen_t{}.model".format(ts)
        model = load_model(model_name)
        train_files, test_files = get_train_test_files()
        test_gen = get_data_gen(files=train_files, timesteps=ts, batch_size=batch_size, im_size=(im_width, im_height))

        y_true = []
        y_pred = []

        for _ in range(200):
            x, y = next(test_gen)
            y_true.extend(y)

            predictions = model.predict_on_batch(x)
            y_pred.extend(predictions)
        psnrs[ts] = [compare_psnr(denormalize(yt), denormalize(p)) for yt, p in zip((y_true), (y_pred))]
        ssims[ts] = [compare_ssim(denormalize(yt), denormalize(p), multichannel=True) for yt, p in zip((y_true), (y_pred))]

    plt.boxplot([psnrs[ts] for ts in timesteps], labels=timesteps)
    plt.savefig("jigsaws_psnrs_all.png")

    plt.figure()
    plt.boxplot([ssims[ts] for ts in timesteps], labels=timesteps)
    plt.savefig("jigsaws_ssims_all.png")

plot_different_models(timesteps=[5, 10])