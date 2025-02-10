# Intelligent Object Recognition and Text Retrieval

**Model Overview:**
Our proposed model is designed as a two-stage pipeline combining a Detection Transformer and an Optical Character Recognition model to efficiently extract textual information from images in real time. The primary objective is to first identify relevant regions containing text using a detection model and then process those regions using OCR model to extract the actual text content. 
In the first stage, we employ a Detection Transformer (DETR), which is retrained on a specific dataset (i,e license plate) from RoboFlow to detect text regions in an image. This model outputs a set of bounding boxes and object labels. Let f_θ be the retrained Detection Transformer model with parameters θ. Given the input image I, the model predicts bounding boxes and object classes i.e. P= f_θ  (I), where P is the set of detected bounding boxes and associated labels. 

The second stage involves an OCR model which takes the detected regions as an input and extracts textual content from them. The OCR model processes each detected region independently, using deep learning techniques to recognize characters and words with high accuracy. Let g_∅ be the Optical Character Recognition (OCR) model with parameters ∅, which takes the detected regions from P and extracts text T i.e. T= g_∅  (P).

Overall, the entire pipeline can be represented as a composite function.
h_(θ,∅)  (I), = g_∅ ( f_θ  (I))
This modular approach ensures efficient text extraction by first localizing textual regions and then applying OCR to those regions, rather than processing the entire image. The model architecture is represented as follow.


