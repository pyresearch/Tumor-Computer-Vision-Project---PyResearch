from flask import Flask, render_template, request, url_for
import os
import cv2
from ultralytics import YOLO
import supervision as sv
import pyresearch

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("last.pt")

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image processing function
def process_image(input_image_path: str, output_image_path: str):
    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image
    resized = cv2.resize(image, (640, 640))

    # Perform detection
    detections = sv.Detections.from_ultralytics(model(resized)[0])

    # Annotate the image
    annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

    # Save the annotated image
    cv2.imwrite(output_image_path, annotated)
    print(f"Processed and saved: {output_image_path}")

# Route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def upload_images():
    processed_images = []  # List to store URLs of processed images
    if request.method == 'POST':
        files = request.files.getlist('files')  # Get multiple files from the form

        for file in files:
            if file and allowed_file(file.filename):
                # Save the uploaded file
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)

                # Define output image path
                output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_' + file.filename)

                # Process the image
                process_image(filename, output_filename)

                # Generate URL for the processed image
                processed_image_url = url_for('static', filename=f'outputs/annotated_{file.filename}')
                processed_images.append(processed_image_url)

    return render_template('index.html', processed_images=processed_images)

if __name__ == "__main__":
    # Create upload and output folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    app.run(debug=True)
