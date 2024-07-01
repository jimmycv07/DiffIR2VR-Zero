import os.path
from skvideo.io import FFmpegWriter
# from image_utils import parse_image_size
# from image_utils import load_image
# from image_utils import resize_image
# from image_utils import list_images_from_dir
from .image_utils import parse_image_size
from .image_utils import load_image
from .image_utils import resize_image
from .image_utils import list_images_from_dir


class VideoVisualizer(object):
    """Defines the video visualizer that presents images as a video."""

    def __init__(self,
                 path=None,
                 frame_size=None,
                 fps=25.0,
                 codec='libx264',
                 pix_fmt='yuv420p',
                 crf=1):
        """Initializes the video visualizer.

        Args:
            path: Path to write the video. (default: None)
            frame_size: Frame size, i.e., (height, width). (default: None)
            fps: Frames per second. (default: 24)
            codec: Codec. (default: `libx264`)
            pix_fmt: Pixel format. (default: `yuv420p`)
            crf: Constant rate factor, which controls the compression. The
                larger this field is, the higher compression and lower quality.
                `0` means no compression and consequently the highest quality.
                To enable QuickTime playing (requires YUV to be 4:2:0, but
                `crf = 0` results YUV to be 4:4:4), please set this field as
                at least 1. (default: 1)
        """
        self.set_path(path)
        self.set_frame_size(frame_size)
        self.set_fps(fps)
        self.set_codec(codec)
        self.set_pix_fmt(pix_fmt)
        self.set_crf(crf)
        self.video = None

    def set_path(self, path=None):
        """Sets the path to save the video."""
        self.path = path

    def set_frame_size(self, frame_size=None):
        """Sets the video frame size."""
        height, width = parse_image_size(frame_size)
        self.frame_height = height
        self.frame_width = width

    def set_fps(self, fps=25.0):
        """Sets the FPS (frame per second) of the video."""
        self.fps = fps

    def set_codec(self, codec='libx264'):
        """Sets the video codec."""
        self.codec = codec

    def set_pix_fmt(self, pix_fmt='yuv420p'):
        """Sets the video pixel format."""
        self.pix_fmt = pix_fmt

    def set_crf(self, crf=1):
        """Sets the CRF (constant rate factor) of the video."""
        self.crf = crf

    def init_video(self):
        """Initializes an empty video with expected settings."""
        assert self.frame_height > 0
        assert self.frame_width > 0

        video_setting = {
            '-r': f'{self.fps:.2f}',
            '-s': f'{self.frame_width}x{self.frame_height}',
            '-vcodec': f'{self.codec}',
            '-crf': f'{self.crf}',
            '-pix_fmt': f'{self.pix_fmt}',
        }
        self.video = FFmpegWriter(self.path, outputdict=video_setting)

    def add(self, frame):
        """Adds a frame into the video visualizer.

        NOTE: The input frame is assumed to be with `RGB` channel order.
        """
        if self.video is None:
            height, width = frame.shape[0:2]
            if height & 1:
                height -= 1
            if width & 1:
                width -= 1
            # height = self.frame_height or height
            # width = self.frame_width or width
            self.set_frame_size((height, width))
            self.init_video()
        if frame.shape[0:2] != (self.frame_height, self.frame_width):
            frame = resize_image(frame, (self.frame_width, self.frame_height))
        self.video.writeFrame(frame)

    def visualize_collection(self, images, save_path=None):
        """Visualizes a collection of images one by one."""
        if save_path is not None and save_path != self.path:
            self.save()
            self.set_path(save_path)
        for image in images:
            self.add(image)
        self.save()

    def visualize_list(self, image_list, save_path=None):
        """Visualizes a list of image files."""
        if save_path is not None and save_path != self.path:
            self.save()
            self.set_path(save_path)
        for filename in image_list:
            image = load_image(filename)
            self.add(image)
        self.save()

    def visualize_directory(self, directory, save_path=None):
        """Visualizes all images under a directory."""
        image_list = list_images_from_dir(directory)
        self.visualize_list(image_list, save_path)

    def save(self):
        """Saves the video by closing the file."""
        if self.video is not None:
            self.video.close()
            self.video = None
            self.set_path(None)


if __name__ == '__main__':
    from glob import glob
    import cv2
    video_visualizer = VideoVisualizer(path='/home/yehhh/DiffBIR/DAVIS_bear.mp4',
                                       frame_size=None,
                                       fps=25.0)
    img_folder = "/home/yehhh/DiffBIR/inputs/bear"
    imgs = sorted(glob(img_folder + '/*.png'))
    imgs = sorted(glob(img_folder + '/*.jpg'))
    for img in imgs:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_visualizer.add(image)
    video_visualizer.save()