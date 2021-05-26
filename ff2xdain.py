import argparse
import time

from ffHelper import FFDestination, FFSource, frameRateMul
from DAINHelper import DAIN2X

parser = argparse.ArgumentParser(prog="FF2xDAIN")
parser.add_argument("-s", "-src", metavar="file_name", dest="src", required=True, help="Source media (file name)")
parser.add_argument("-d", "-dst", metavar="file_name", dest="dst", required=True, help="Destination (file name)")
parser.add_argument("-ffsp", dest="ffsp", help="Custom ffmpeg source parameters")
parser.add_argument("-ffdp", dest="ffdp", help="Custom ffmpeg destination parameters")

if __name__ == "__main__":
    args = parser.parse_args()

    try:
        dain2x = DAIN2X("./model_weights/best.pth")
        src = FFSource(args.src, param=args.ffsp.split(" ") if args.ffsp is not None else [])
        with FFDestination(args.dst,
                        src.streamInfo["width"],
                        src.streamInfo["height"],
                        frameRateMul(src.streamInfo["r_frame_rate"], 2),
                        param=args.ffdp.split(" ") if args.ffdp is not None else []) as dest:
            
            n = 0
            for frame in src:
                n += 1
                time_stamp = time.time()
                print(f"frame: {n} processing...")

                iframe = dain2x(frame)
                print(f"frame: {n} done proc. Time: {(time.time()-time_stamp):.2f}")
                if iframe is not None:
                    dest.putData(iframe)

                dest.putData(frame)
            
            print(f"Frames total:{n}")
        
    except Exception as e:
        print(e)
        exit(-1)
