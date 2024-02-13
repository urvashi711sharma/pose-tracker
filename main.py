import cv2

# Load pre-trained OpenPose model
net = cv2.dnn.readNetFromTensorflow("path/to/openpose/model.pb")

# Specify the input image size expected by the model
inWidth = 368
inHeight = 368

# Define the set of points corresponding to the joints of the human body
joint_points = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]

# Update the file path to point to the Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('C:/a/pose tracker/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from camera")
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Prepare the input image for OpenPose model
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared blob as input to the network
    net.setInput(blob)

    # Forward pass through the network to get the output
    out = net.forward()

    # Get the height and width of the frame
    frame_height, frame_width = frame.shape[:2]

    # Iterate through detected points and draw lines on joints
    for i in range(len(joint_points)):
        # Extract the x, y coordinates of the joint
        joint_x = int(out[0, i, 0] * frame_width)
        joint_y = int(out[0, i, 1] * frame_height)

        # Draw circles at the joint points
        cv2.circle(frame, (joint_x, joint_y), 5, (0, 255, 255), -1)

        # Put text label at the joint point
        cv2.putText(frame, joint_points[i], (joint_x, joint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Iterate through detected faces and draw rectangles around them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
