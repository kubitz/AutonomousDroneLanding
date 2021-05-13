"""
This scripts allows to generate the config file required to run the C++ code.
It creates two/three files:
    * inputs.txt: contains all the images to be inferred - this will be all images present in the data/imgs folder (except if SIMULATION is on)
    * masks.txt (for SIMULATION=True only): paths of all image masks
    * weights.txt (for SIMULATION=false only): path for the weight files
"""

SIMULATE = False # Change this value to generate


from pathlib import Path
import glob
from configparser import ConfigParser
# TODO: Add argparse to chose what folders to generate the config for

basePath = Path.cwd().parents[0]
dataPath = str(basePath.joinpath("data", "imgs"))
imgsCfgPath = str(basePath.joinpath("data", "cfg", "inputs.cfg"))

dataFolders = []

for path in Path(dataPath).iterdir():
    if path.is_dir():
        dataFolders.append(path)
dataFolders.sort()

if SIMULATE:
    segSimPath = str(basePath.joinpath("data", "imgs", dataFolders[0].stem, "masks", "*"))
    imgPath = str(basePath.joinpath("data", "imgs", dataFolders[0].stem, "images", "*"))
    segSimCfgPath = str(basePath.joinpath("data", "cfg", "masks.cfg"))
    masksImgs = glob.glob(segSimPath)
    inputImgs = glob.glob(imgPath)
    masksImgs.sort()
    inputImgs.sort()

    with open(segSimCfgPath, 'w') as f:
        for mask in masksImgs:
            f.write("%s\n" % str(mask))

    with open(imgsCfgPath, 'w') as f:
        for img in inputImgs:
            f.write("%s\n" % str(img))
else:
    # TODO: add exception/Warning if weight files have not been downloaded yet
    dataFolders.pop(0)
    weightObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "yolov3_leaky.weights")
    )
    cfgObjPath = str(basePath.joinpath("data", "weights", "yolo-v3", "yolov3_leaky.cfg"))
    namesObjPath = str(
        basePath.joinpath("data", "weights", "yolo-v3", "visdrone.names")
    )
    weightSegPath = str(
        basePath.joinpath("data", "weights", "seg", "Unet-Mobilenet.pt")
    )
    weightCfgPath = str(basePath.joinpath("data", "cfg", "weights.cfg"))
    with open(weightCfgPath, 'w') as f:
        f.write("%s\n" % weightObjPath)
        f.write("%s\n" % cfgObjPath)
        f.write("%s\n" % namesObjPath)
        f.write("%s\n" % weightSegPath)
    imgsSequences = []
    for folder in dataFolders:
        imgPath = str(basePath.joinpath("data", "imgs", folder.stem, "images", "*"))
        inputImgs = glob.glob(imgPath)
        inputImgs.sort()
        imgsSequences += inputImgs

    with open(imgsCfgPath, 'w') as f:
        for img in imgsSequences:
            f.write("%s\n" % str(img))
