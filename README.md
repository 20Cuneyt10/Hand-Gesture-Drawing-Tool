# Hand-Gesture-Drawing-Tool
This is a Python-based desktop application that uses your webcam to create an interactive drawing tool. It leverages the MediaPipe library for real-time hand tracking and gesture recognition. The application allows you to "draw" on the screen by simply moving your hand.

How It Works

The app captures video from your webcam and analyzes each frame to identify the position of your hands and the key landmarks on your fingers. It uses a pinch gesture (bringing your thumb and index finger together) to start and stop drawing. Once a pinch is detected, the application tracks the tip of your index finger and draws a colored line that follows its movement.

The lines are color-coded: green for your left hand and red for your right hand. To clear the drawing, you can make a fist with either hand. The code includes smoothing to make the drawn lines appear more fluid and natural. You can mess with the touch_treshold for the distance between the two fingers to do the drawing or mess with the smoothing alpha for faster tracking.
