from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import cv2
import torch

from ClassYoloDetection import YoloDetection

'''
    This class tries to track peoples locations in a video. It does this by running a person detection model (using
    yolo) on each frame, taking the detected bounding boxes and embedding them using the embedding layer of the 
    CLIP model. These embeddings can then be compared with embeddings from previous frames, which allows us to 
    identify a detected person.
    
    This method is privacy friendly, because you don't need to save the camera images, only the embedded layers, which 
    (after being embedded) can no longer be changed back towards a camera image.
    
    NOTE: There is still a lot that can be improved on this, such as filtering out locations, comparing embedded layers
    with multiple previous frames, comparing camera images from different cameras, etc. This is more of a jumping
    of point and a different way of tracking people that does not rely on storing images of people.
    
    NOTE: Because of the way I defined self.ID_colors, this should only work with up to 10 people in a single frame (though
    I have only tested it with 2). Use at your own discretion.

    Author :        Martijn Folmer 
    Date created :  24-06-2023
'''


class SimilarityTracking:
    def __init__(self):
        print("Initialized the Similarity Tracking")

        # initialize the class that does the yolo object detection
        self.YD = YoloDetection('YOLOmodels/yolov7-tiny_256x480.onnx', class_idx_to_detect=[0],
                                confidence_threshold=0.7, iou_threshold=0.2)

        # all the colors of IDS (meaning which colors we use to indicate different IDs/different persons)
        self.ID_colors = [[255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 128], [128, 0, 0],
                          [128, 128, 0], [128, 0, 128], [0, 128, 128], [0, 0, 128]]

        self.all_ids = []  # Consists of [[list of encoded_images], list of [Frame_num, xc, yc, x1, y1, x2, y2] of where we are been]
        self.maxNumberEncoded = 1   # The amount of embedded images from the past N frames we store
        self.sim_threshold = 0.8    # The threshold for which we say, it needs to be higher than that to be a tracking match

        print('Load the clip model : ')
        self.Clip_model = SentenceTransformer('clip-ViT-B-32')      # we need the Clip model to embed the image

    def EncodeImage(self, _frame):
        """
        This function encodes the images into an embedded layer that we can use to compare to eachother

        :param _frame: the image we want to encode (usually the bounding box of a person we cut out)
        :return: the encoded image.
        """
        _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(_frame)                # turn it into a PIL image format, for encoding
        encoded_image = self.Clip_model.encode([im_pil], batch_size=1, convert_to_tensor=True)  # the encoded image is 1x512, because that is the input size of the CLIP pytorch tensor
        return encoded_image

    def cutOutBB(self, _frame, _box):
        """
        Take a bounding box of a person, and cut out that specific part of the image

        :param _frame: The full image ran the detection on
        :param _box: the bounding box we detected [x1, y1, x2, y2]
        :return: The cut out part of the image (so only what is within the bounding box)
        """
        h, w, _ = _frame.shape
        x1, y1, x2, y2 = _box
        x1 = max(min(w, x1), 0)
        y1 = max(min(h, y1), 0)
        x2 = max(min(w, x2), 0)
        y2 = max(min(h, y2), 0)

        return _frame[int(y1):int(y2), int(x1):int(x2)]

    def ReadVideo(self, pathToVideo):
        """
        This is the main function of the class, which takes the path to a video, reads each frame and plots the
        paths of the detected people in two sepperate videos under results.avi (which shows the
         bounding boxes) and results_track.avi (which shows the x and y center of each detected person over time.

        :param pathToVideo: The path to the video we want to read and detect people from.
        :return: It saves two videos, the results.avi and the results_track.avi
        """
        cap = cv2.VideoCapture(pathToVideo)
        res, frame = cap.read()     # first frame, which we read in order to find the frame
        frame_w, frame_h = frame.shape[1], frame.shape[0]       # Get the shape of the frames

        # open the video we want to write too
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter('results.avi', fourcc, 30.0, (frame_w, frame_h))
        amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        kn = 0
        while True:
            res, frame = cap.read()

            # If we are out of frames, break out
            if not res:
                break

            # Run the YOLO detection
            boxes, scores, class_ids = self.YD.RunObjectDetection(frame)

            # check them for our previous grids
            for box_c in boxes:
                cutout = self.cutOutBB(frame, box_c)
                cutout_embedded = self.EncodeImage(cutout)

                # get the tensors of all the ids
                FoundMatch = False
                allEncoded = cutout_embedded
                for i_idx, ids in enumerate(self.all_ids):
                    allEncoded = torch.cat([allEncoded, ids[0][0]], dim=0)
                if len(allEncoded) > 1:
                    processed_images = util.paraphrase_mining_embeddings(allEncoded)  # first element is always highest

                    # check if the similarity between the two embedded layers exceeds the similarity threshold
                    if processed_images[0][0]>self.sim_threshold:

                        # We find the index which is not 0, which is the id of the person we match with
                        idx1, idx2 = processed_images[0][1], processed_images[0][2]
                        if idx1 == 0:
                            idx = idx2-1
                        else:
                            idx = idx1-1

                        FoundMatch = True           # we found a match to track
                        self.all_ids[idx][1].append([kn, (box_c[0]+box_c[2])/2, (box_c[1]+box_c[3])/2, box_c[0], box_c[1], box_c[2], box_c[3]])
                        self.DrawBoxColor(frame, box_c, self.ID_colors[idx], idx)    # draw the bounding box with colors

                        # update the embedded every so often
                        if len(self.all_ids[idx][1]) % self.maxNumberEncoded == 0 and len(self.all_ids[idx][1])>1:
                            self.all_ids[idx][0] = [cutout_embedded]

                # We did not find a match, so this must be a new person
                if not FoundMatch:
                    self.all_ids.append([[cutout_embedded], [[kn, (box_c[0]+box_c[2])/2, (box_c[1]+box_c[3])/2, box_c[0], box_c[1], box_c[2], box_c[3]]]])
                    self.DrawBoxColor(frame, box_c, self.ID_colors[len(self.all_ids)], len(self.all_ids))        # Draw the bounding box with colors

            # Write to our video
            video_out.write(frame)
            kn += 1
            print(f"We are at frame {kn} / {int(amount_of_frames)}, which is {int(100*kn/amount_of_frames)}%")
            
        # Set our found tracking coordinates per frame
        trackingTotal = []
        for i_idcur, idcur in enumerate(self.all_ids):
            curTrack = [[0, 0] for _ in range(kn)]
            for loc in idcur[1]:
                curTrack[loc[0]] = [loc[1], loc[2]]
            trackingTotal.append(curTrack)

        # Create the BG tracking
        video_bg_out = cv2.VideoWriter('results_track.avi', fourcc, 30.0, (frame_w, frame_h))
        for frame_num in range(len(trackingTotal[0])):
            bg_track = np.zeros((frame_h, frame_w, 3))
            for i in range(frame_num):
                for i_track, track in enumerate(trackingTotal):
                    x1, y1 = track[i]
                    if x1 != 0 and y1 != 0:
                        if i == frame_num-1:
                            bg_track = cv2.circle(bg_track, (int(x1), int(y1)), 17, (255, 255, 255), -1)
                            bg_track = cv2.circle(bg_track, (int(x1), int(y1)), 15, (0, 0, 0), -1)
                            bg_track = cv2.circle(bg_track, (int(x1), int(y1)), 10, self.ID_colors[i_track], -1)

                            bg_track = self.DrawTextWithOutline(bg_track, (int(x1), int(y1-30)), f"Person id : {i_track}", self.ID_colors[i_track], (255, 255, 255))

                        else:
                            bg_track = cv2.circle(bg_track, (int(x1), int(y1)), 4, self.ID_colors[i_track], -1)

            video_bg_out.write(np.asarray(bg_track, dtype=np.uint8))

        # cleanup
        cap.release()
        video_out.release()
        video_bg_out.release()
        cv2.destroyAllWindows()

    # Visualisation functions
    def DrawTextWithOutline(self, _frame, _org, _txt, _color_inside, _color_outline):

        _frame = cv2.putText(_frame, _txt, (int(_org[0] + 2), int(_org[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             _color_outline, 2,
                             cv2.LINE_AA)
        _frame = cv2.putText(_frame, _txt, (int(_org[0] - 2), int(_org[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             _color_outline, 2,
                             cv2.LINE_AA)
        _frame = cv2.putText(_frame, _txt, (int(_org[0]), int(_org[1] + 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             _color_outline, 2,
                             cv2.LINE_AA)
        _frame = cv2.putText(_frame, _txt, (int(_org[0]), int(_org[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                             _color_outline, 2,
                             cv2.LINE_AA)

        _frame = cv2.putText(_frame, _txt, (int(_org[0]), int(_org[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, _color_inside, 2,
                             cv2.LINE_AA)
        return _frame

    def DrawBoxColor(self, _frame, _box, _color, _idx):
        x1, y1, x2, y2 = _box  # Get the center coordinates and width and height
        _frame = cv2.rectangle(_frame, (int(x1 - 2), int(y1 - 2)), (int(x2 + 2), int(y2 + 2)), _color, 8)

        _frame = self.DrawTextWithOutline(_frame, (int(x1 + 20), int(y1 + 20)), f"Person id : {_idx}", _color,
                                          (255, 255, 255))

        return _frame


if __name__ == "__main__":
    pathToVideo = 'Path/To/The/Video/We/Want/To/Run'
    ST = SimilarityTracking()
    ST.ReadVideo(pathToVideo=pathToVideo)
