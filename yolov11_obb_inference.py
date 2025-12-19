from ultralytics import YOLO
import matplotlib.pyplot as plt



# Load a model
# model = YOLO(r"C:\Users\victa\OneDrive\Documents\GitHub\2025_Hackathon\trained_models\yolov11_obb\model_1\weights\best.pt")  # load a pretrained model (recommended for training)
# model = YOLO(r"C:\Users\victa\OneDrive\Documents\GitHub\2025_Hackathon\trained_models\yolov11_obb\model_2\runs\obb\train2\weights\best.pt")
model = YOLO(r"C:\Users\victa\OneDrive\Documents\GitHub\2025_Hackathon\trained_models\yolov11_obb\model_4\runs\obb\train2\weights\best.pt")
real_im = r"C:\Users\victa\OneDrive\Documents\GitHub\2025_Hackathon\CdTe_Training_Image_photoshop_fake_defects_TP.png"

# Train the model
results = model(real_im, show=True)


# Render image with predictions (OBB included)
img = results[0].plot()   # returns a numpy array (BGR)

# Convert BGR → RGB for matplotlib
img = img[:, :, ::-1]

# Display
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()