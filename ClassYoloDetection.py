import cv2
import numpy as np
import onnxruntime

'''
    This class uses a pre trained yolo-network in order to perform basic Object Detection.
    
    I use a yolov7-tiny_256x480.onnx, which was downloaded from PINTO0309 at the following repository:
    https://github.com/PINTO0309/PINTO_model_zoo/tree/main/307_YOLOv7
    
    Other .onnx models you download from this repository should work as well, as they have the same format

    Author :        Martijn Folmer 
    Date created :  24-06-2023
'''


class YoloDetection:
    def __init__(self, pathToYolo, class_idx_to_detect=[0], confidence_threshold=0.7, iou_threshold=0.3):

        self.initializeYoloModel(pathToYolo)            # Initialize the yolo format
        self.iou_threshold = iou_threshold              # how much the boxes must be overlapping to be removed during NMS
        self.conf_threshold = confidence_threshold      # the minimum confidence score of the detected objects
        self.class_idx_to_detect = class_idx_to_detect  # the types of objects we are looking for. So if you pass on [0], that means we will only look for self.class_names[0], which is 'person'

        # All the class names for the types of objects that a standard yolo model can detect
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def initializeYoloModel(self, modelPath):
        """
        Initialize the yolo model, get all of the tensor names and input shape that we need to run it

        :param modelPath: The path to the .onnx file where we stored the yolo
        :return: --
        """

        self.session = onnxruntime.InferenceSession(modelPath, providers=['CPUExecutionProvider'])
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"Input shape of the yolo model is : {self.input_shape}")

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


    def preProcessImg(self, image):
        """
        Perform all of the pre-processing steps to run an image through the yolo network

        :param image: the array representing the image (can be loaded using cv2)
        :return: The tensor we can run through the yolo network
        """

        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        """
        Run the yolo network on the input tensor

        :param input_tensor: The input tensor, which is a representation of the image we want to pass through the yolo
        :return: The outputs from the yolo model
        """
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def RunObjectDetection(self, image):
        """
        Perform the entire object detection, by taking an image, preprocessing it, running the yolo network and postprocessing
        the output

        :param image: The loaded image we want to run through the yolo network
        :return: the outputs of the yolo model (boxes, confidence_scores, class_ids)
        """

        # Preprocess the image and
        input_tensor = self.preProcessImg(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        self.boxes, self.scores, self.class_ids = self.postProcessOutput(outputs)

        return self.boxes, self.scores, self.class_ids

    def postProcessOutput(self, output):
        """
        Take the output of the yolo model, and perform all post processing, such as confidence threshold and non
        max suppression

        :param output: the output from the yolo network
        :return: boxes ([[x1, y1, x2, y2],....,[xn-1, yn-1, xn, yn]], scores [0-1], class_ids [int]
        """

        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.GetBoxesFromOutput(predictions)

        # Non-max suppression, in order to get rid of overlapping
        boxes, scores, class_ids, _ = self.non_max_suppression(boxes, scores, class_ids, self.iou_threshold)

        # Filter out all the ids that we don't want to track
        boxesFiltered, scoresFiltered, class_idsFiltered = [], [], []
        for (boxCur, scoreCur, class_idsCur) in zip(boxes, scores, class_ids):
            if class_idsCur in self.class_idx_to_detect:
                boxesFiltered.append(boxCur)
                scoresFiltered.append(scoreCur)
                class_idsFiltered.append(class_idsCur)

        return boxesFiltered, scoresFiltered, class_idsFiltered

    def GetBoxesFromOutput(self, predictions):
        """
        Take the predictions from the output of the yolo network, find the boxes (xcenter, ycenter, width, height),
        rescale them according to original image dimensions and turn them into format [x1, y1, x2, y2].

        :param predictions: The output from the yolo model
        :return: the bounding boxes [x1, y1, x2, y2]
        """
        # Get the boxes
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.RescaleBoxes(boxes)

        # Turn boxes to [x1, y1, x2, y2] from [xcenter, ycenter, width, height]
        boxes = [[box[0]-box[2]/2, box[1] - box[3]/2, box[0]+box[2]/2, box[1] + box[3]/2] for box in boxes]

        return boxes


    def RescaleBoxes(self, boxes):
        """
        The image we use as input for the yolo network does not have to have the same input size as the input shape
        of the model, so we have to make sure that the output is appropriately scaled for the size of the input image

        :param boxes: [[xc, yc, width, height], ...,]
        :return: rescaled boxes.
        """
    
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height]) # the input of the model
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])       # the original image size
        return boxes

    def isBoxAinBoxB(self, boxA, boxB):
        """
        During Non-max suppression, we check whether one box is completely engulfed in another box

        :param boxA: the corner coordinates of the bounding box [x1, y1, x2, y2]
        :param boxB: the corner coordinates of the bounding box [x3, y3, x4, y4]
        :return: True if boxA is inside of boxB, False if not
        """
        
        x1, y1, x2, y2 = boxA
        x3, y3, x4, y4 = boxB

        if x3 < x1 < x4 and y3 < y1 < y4:
            if x3 < x2 < x4 and y3 < y2 < y4:
                return True
        return False

    def IntersectionOverUnion(self, boxA, boxB):
        """
        Perform the Intersection Over Union algorithm, which checks how much boxes overlap with eachother
        
        :param boxA: 
        :param boxB: 
        :return: between 0 and 1, where 0 is no overlap, 1 is overlap
        """
        intersection_x1 = max(boxA[0], boxB[0])
        intersection_y1 = max(boxA[1], boxB[1])
        intersection_x2 = min(boxA[2], boxB[2])
        intersection_y2 = min(boxA[3], boxB[3])

        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0,
                                                                            intersection_y2 - intersection_y1)
        current_area = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        selected_area = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # Calculate the IoU (Intersection over Union)
        denom = float(current_area + selected_area - intersection_area)
        if denom == 0:
            denom = 0.001
        iou = intersection_area / denom

        return iou

    def non_max_suppression(self, bounding_boxes, confidence_scores, all_class_idx, overlap_threshold):
        """
        Non max suppresion is an algorithm to take multiple bounding boxes that overlap, and average them out in a
        way so you are only left with the best bounding box for any given space

        :param bounding_boxes: The list of the bounding boxes
        :param confidence_scores: The list of all the confidence scores (between 0 and 1)
        :param all_class_idx: The id of the type of object we found inside of the bounding box
        :param overlap_threshold: If overlap exceeds this number, we get rid of the lower confidence
        :return: the selected bounding boxes, confidence scores, their ids and their indexes
        """
        # Sort the bounding boxes based on their areas (or scores)
        sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: confidence_scores[i], reverse=True)

        # Initialize an empty list to store the selected bounding boxes
        boxBes = []
        boxBes_idx = []

        # Iterate over the sorted bounding boxes
        for i in range(len(sorted_indices)):
            boxA = bounding_boxes[sorted_indices[i]]
            should_select = True

            # Check if the current bounding box overlaps significantly with any selected bounding boxes
            for j in range(len(boxBes)):
                if should_select:
                    boxB = boxBes[j]

                    boxinbox = (self.isBoxAinBoxB(boxB, boxA) or self.isBoxAinBoxB(boxA,boxB))
                    iou = 0
                    if boxinbox == False:
                        iou = self.IntersectionOverUnion(boxA, boxB)

                    # Discard the current bounding box if the IoU exceeds the threshold
                    if iou > overlap_threshold or boxinbox:
                        # Leaky Non-max-suppression : alter the coordinates slightly based on other confident boxes
                        should_select = False

            # Add the current bounding box to the selected list
            if should_select:
                boxBes.append(bounding_boxes[sorted_indices[i]])
                boxBes_idx.append(sorted_indices[i])

        selected_confidence = [confidence_scores[idx] for idx in boxBes_idx]
        selected_class_idx = [all_class_idx[idx] for idx in boxBes_idx]

        return boxBes, selected_confidence, selected_class_idx, boxBes_idx


if __name__ == "__main__":

    # The following code will run an image through the yolo object detectition model and return an image with bounding
    # boxes around each person detected in the imag

    PathToImg = 'Path/To/The/Image/You/Want/To/Run/Yolo/On'
    YD = YoloDetection('YOLOmodels/yolov7-tiny_256x480.onnx', class_idx_to_detect=[0], confidence_threshold=0.7,
                       iou_threshold=0.3)
    img = cv2.imread(PathToImg)
    cv2.imshow("Detected People", img)

    # run the detection
    boxes, scores, class_ids = YD.RunObjectDetection(img)

    # Visualisation, draw the boxes on our desired image.
    for (box, score, id) in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box      # Get the center coordinates and width and height
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))

    cv2.imshow('people detection', img)
    cv2.waitKey(-1)
