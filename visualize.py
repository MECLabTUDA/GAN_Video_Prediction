from data_utils import get_data_gen, get_train_test_files, denormalize, VIDEO_KNOT, VIDEO_NEEDLE_PASSING, VIDEO_SUTURING
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set_session(sess)  # s

print("Loading model...")
model = load_model("./r_p2p_gen.model")
print("Loaded model")

def plot_results(y_true, y_pred, filename, cols=5):
    rows = max(1, len(y_true) // cols * 2)
    # draw actuals in first row
    # then predicted in 2nd row and so on
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(1*cols, 1*rows))
    actual_index = 0
    predicted_index = 0
    for i in range(rows):
        for j in range(cols):

            if i % 2 == 0:
                img = y_true[actual_index]
                actual_index += 1
            else:
                img = y_pred[predicted_index]
                predicted_index += 1

            axes[i, j].imshow(img, interpolation="none")
            if j == 0:
                axes[i, j].set_ylabel("Actual" if i % 2 == 0 else "Predicted")
            axes[i, j].get_xaxis().set_ticks([])
            axes[i, j].get_yaxis().set_ticks([])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(filename, bbox_inches='tight')

def plot_metrics(y_true, y_pred):
    from skimage.measure import compare_psnr, compare_ssim
    psnrs = [compare_psnr(yt, p) for yt, p in zip(denormalize(y_true), denormalize(y_pred))]
    ssims = [compare_ssim(yt, p, multichannel=True) for yt, p in zip(denormalize(y_true), denormalize(y_pred))]
    plt.figure(figsize=(5, 4))
    plt.boxplot(psnrs, 0, 'gD')
    plt.savefig("./jigsaws_psnrs_boxplot.png")

    plt.figure(figsize=(5, 4))
    plt.boxplot(ssims, 0, 'rD')
    plt.savefig("./jigsaws_ssims_boxplot.png")

    print("Mean PSNR = ", np.mean(np.array(psnrs)))
    print("Mean SSIM = ", np.mean(np.array(ssims)))

def plot_saliency(model, y_true):
    def visualize_filters(model, layer_name, input_data, filter_indices=None, mode="guided"):
        from vis.visualization import get_num_filters, visualize_saliency
        from vis.utils import utils
        from vis.input_modifiers import Jitter
        """
        Visualize what pattern activates a filter. Helps to discover what a 
        filter might be computing
        :returns tuple(List, List) containing input images and heatmaps
                frames from each sample is stitched into a single image
        """
        get_num_filters
        inputs = []
        outputs = []
        # number of filters for this layer
        num_filters = get_num_filters(model.get_layer(layer_name))
        layer_idx = utils.find_layer_idx(model, layer_name)    
        for sample in input_data:
            heatmaps = visualize_saliency(model, layer_idx, filter_indices=filter_indices, 
                                        seed_input=sample, 
                                        backprop_modifier=mode,
                                        )
            inputs.append(utils.stitch_images(sample, margin=0))
            outputs.append(utils.stitch_images(heatmaps, margin=0))
            
        return np.array(inputs), np.array(outputs)

    layers_to_visualize = ["conv_lst_m2d_2", "conv2d_19"]
    input_data = y_true
    visualizations = {}
    for i, l in enumerate(layers_to_visualize):
        mode = "relu" if i < len(layers_to_visualize) - 1 else "guided"
        print("Visualizing layer {} using mode {}".format(l, mode))
        # guided for last layer, relu for others
        
        inputs, outputs = visualize_filters(model, layer_name=l, input_data=input_data, mode=mode)
        visualizations[l] = (inputs, outputs)

    num_vis = len(layers_to_visualize)+1
    num_rows = (len(input_data)*num_vis) // 2
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, squeeze=False, figsize=(15, 10))
    for i in range(0, len(input_data)*num_vis, num_vis):
        data_idx = i // num_vis
        col_idx = 0 if data_idx % 2 == 0 else 1
        row_start_idx = 0
        if i == 0 or i == 3:
            row_start_idx = 0
        elif i == 6 or i == 9:
            row_start_idx = 3
        elif i == 12 or i == 15:
            row_start_idx = 6
        for j, layer_name in enumerate(layers_to_visualize):
            inp, op = visualizations[layer_name]
            if j == 0:
                axes[row_start_idx+j, col_idx].imshow(inp[data_idx], aspect="auto")
                axes[row_start_idx+j+1, col_idx].imshow(op[data_idx], aspect="auto")
                
                if col_idx == 0:
                    axes[row_start_idx+j, col_idx].set_ylabel("Input")
                    axes[row_start_idx+j+1, col_idx].set_ylabel("1st Layer")
            else:
                axes[row_start_idx+j+1, col_idx].imshow(op[data_idx], aspect="auto")
                if col_idx == 0:
                    axes[row_start_idx+j+1, col_idx].set_ylabel("Last Layer")
                
    #         axes[i, j].set_ylabel("Actual" if i % 2 == 0 else "Predicted")

    for a in axes.flatten():
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
    fig.subplots_adjust(wspace=0.01, hspace=0.0)   
    plt.savefig("./jigsaws_saliency.png", bbox_inches='tight', pad_inches=0.0)

def visualize_for(which, filename, should_plot_results=True, should_plot_metrics=False, should_plot_saliency=False):
    train_files, test_files = get_train_test_files(shuffle=True, which=which)
    test_gen = get_data_gen(files=test_files, timesteps=timesteps, 
                        batch_size=batch_size, 
                        im_size=(im_width, im_height),
                        fps=5                    
                        )
    if which is not None:
        next(test_gen)
    
    x, y_true = next(test_gen)
    # for small
    # x, y_true = x[-5:], y_true[-5:]
    y_pred = model.predict(x, batch_size=5)

    if should_plot_results:
        plot_results(y_true, y_pred, filename=filename, cols=10)

    if should_plot_metrics:
        plot_metrics(y_true, y_pred)

    if should_plot_saliency:
        p = np.random.permutation(len(x))
        plot_saliency(model, x[p][:4])


batch_size = 20
timesteps = 5
im_width = im_height = 256

# visualize_for(VIDEO_KNOT, "knot_predictions.png")
# visualize_for(VIDEO_NEEDLE_PASSING, "needle_predictions.png")
# visualize_for(VIDEO_SUTURING, "suturing_predictions.png")

batch_size = 600
visualize_for(None, "", should_plot_results=False, should_plot_metrics=True, should_plot_saliency=False)
