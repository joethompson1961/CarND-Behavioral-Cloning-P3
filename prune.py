import csv
from shutil import copy2
import errno    
import os

what = "simdata"
#what = "tunedata"

src_dir = '..\\' + what + '\\'
dest_dir = '..\\' + what + '_pruned\\'

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# make directories for pruned data
mkdir_p(dest_dir)
mkdir_p(dest_dir + 'IMG')

# make a copy of the data log file
copy2(src_dir +'driving_log.csv', dest_dir)

lines = []
with open(src_dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Copy the images referenced in the data log file from the original location to the pruned directory.
# This copies only the reference images and leaves the unreferenced images in the original location.
dest = dest_dir + 'IMG\\'
for line in lines:
    source_path = line[0]
    fn = source_path.split('\\')[-1]
    print('copying:' + fn)
    src = src_dir + 'IMG\\' + fn

    copy2(src, dest)
