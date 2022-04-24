from pathlib import PurePath
from sys import argv
import ffmpeg
from os import listdir, remove
import re

def create_video(foldername, framerate = 3):
    files = [f for f in listdir(foldername) if re.match(r'[0-9]{4}\.png', f)]
    files.sort()

    with open(PurePath(foldername, 'images.txt'), 'w', encoding='utf8') as f:
        for name in files:
            f.write(f'file \'{name}\'\n')

    
    ffmpeg.input(PurePath(foldername, 'images.txt'), r=framerate, f='concat') \
        .output(f'{foldername}/animation.mp4', pix_fmt='yuv420p', vcodec='libx264').run()

    remove(PurePath(foldername, 'images.txt'))

if __name__ == '__main__' and len(argv) > 1:
    create_video(argv[1])