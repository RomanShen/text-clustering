from PIL import Image
import os


def generate_gif(dir_path, gif_path, duration=500):
    """
    :param dir_path: the directory path of images, named in sequential order, i.e., 1, 2, 3, 4...
    :param duration: interval between two frames
    :param gif_path: path of the returned gif
    :return: gif
    """
    images = []
    filenames = sorted(fn for fn in os.listdir(dir_path) if fn.endswith('png'))
    imN = 1
    for filename in filenames:
        if imN == 1:
            im = Image.open(dir_path + filename)
            imN = 2
            images.append(Image.open(dir_path + filename))
    im.save(gif_path, save_all=True, append_images=images, loop=50, duration=duration)

