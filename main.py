import pygame
import math
import cv2 as cv
import mediapipe as mp
import time

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MediaPipe Drawing Tool")

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
font = pygame.font.SysFont(None, 24)

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_near_head_start_time = None
QUIT_HOLD_DURATION_SEC = 1.0 

drawing_mode_left = False
drawing_mode_right = False
DRAWING_TOGGLE_COOLDOWN_SEC = 0.5
last_drawing_toggle_time = 0

DRAWING_BREAK_DURATION_SEC = 1.0
last_hand_detected_time = time.time()
SMOOTHING_ALPHA = 0.3
smoothed_positions = {'left': None, 'right': None}

try:
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open webcam")
    
    cam.set(cv.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.4
    )
except Exception as e:
    print(f"Error initializing webcam or MediaPipe: {e}")
    pygame.quit()
    exit()

running = True
drawing_points = {'left': [], 'right': []}

def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False
    
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    mcp_joints = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    for tip, mcp in zip(finger_tips, mcp_joints):
        tip_y = hand_landmarks.landmark[tip].y
        mcp_y = hand_landmarks.landmark[mcp].y
        if tip_y < mcp_y:
            return False
            
    return True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    success, frame = cam.read()
    if not success:
        continue
    
    frame_height, frame_width, _ = frame.shape
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    
    current_time = time.time()
    
    drawing_hands = {'left': None, 'right': None}
    hand_is_visible = False

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        hand_is_visible = True
        last_hand_detected_time = current_time
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label
            if label == 'Left':
                drawing_hands['left'] = hand_landmarks
            elif label == 'Right':
                drawing_hands['right'] = hand_landmarks
                
    should_continue_drawing = (current_time - last_hand_detected_time) < DRAWING_BREAK_DURATION_SEC

    for hand_label in ['left', 'right']:
        hand_landmarks = drawing_hands[hand_label]
        
        if hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            if is_fist(hand_landmarks):
                drawing_points[hand_label].clear()
            else:
                try:
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    drawing_distance = math.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2
                    ) * frame_width
                    
                    if drawing_distance < 40:
                        if (current_time - last_drawing_toggle_time) > DRAWING_TOGGLE_COOLDOWN_SEC:
                            if hand_label == 'left':
                                drawing_mode_left = True
                            else:
                                drawing_mode_right = True
                            last_drawing_toggle_time = current_time
                    else:
                        if hand_label == 'left':
                            drawing_mode_left = False
                        else:
                            drawing_mode_right = False
                    
                    if (hand_label == 'left' and drawing_mode_left) or (hand_label == 'right' and drawing_mode_right):
                        current_pos = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
                        
                        if smoothed_positions[hand_label] is None:
                            smoothed_positions[hand_label] = current_pos
                        else:
                            smoothed_positions[hand_label] = (
                                smoothed_positions[hand_label][0] * (1 - SMOOTHING_ALPHA) + current_pos[0] * SMOOTHING_ALPHA,
                                smoothed_positions[hand_label][1] * (1 - SMOOTHING_ALPHA) + current_pos[1] * SMOOTHING_ALPHA
                            )
                        drawing_points[hand_label].append(smoothed_positions[hand_label])
                    else:
                        if drawing_points[hand_label] and drawing_points[hand_label][-1] is not None:
                            drawing_points[hand_label].append(None)
                except IndexError:
                    pass
        elif should_continue_drawing and ((hand_label == 'left' and drawing_mode_left) or (hand_label == 'right' and drawing_mode_right)):
            if drawing_points[hand_label] and drawing_points[hand_label][-1] is not None:
                drawing_points[hand_label].append(None)

    if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        if len(hand_results.multi_hand_landmarks) == 2:
            hand1_landmarks = hand_results.multi_hand_landmarks[0]
            hand2_landmarks = hand_results.multi_hand_landmarks[1]
            nose_tip = face_results.multi_face_landmarks[0].landmark[1]
            nose_pos = (int(nose_tip.x * frame_width), int(nose_tip.y * frame_height))
            hand1_tip = hand1_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand2_tip = hand2_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand1_tip_pos = (int(hand1_tip.x * frame_width), int(hand1_tip.y * frame_height))
            hand2_tip_pos = (int(hand2_tip.x * frame_width), int(hand2_tip.y * frame_height))
            dist1 = math.sqrt((hand1_tip_pos[0] - nose_pos[0])**2 + (hand1_tip_pos[1] - nose_pos[1])**2)
            dist2 = math.sqrt((hand2_tip_pos[0] - nose_pos[0])**2 + (hand2_tip_pos[1] - nose_pos[1])**2)
            touch_threshold = 120

            if dist1 < touch_threshold and dist2 < touch_threshold:
                if hands_near_head_start_time is None:
                    hands_near_head_start_time = time.time()
                elif (time.time() - hands_near_head_start_time) > QUIT_HOLD_DURATION_SEC:
                    running = False
            else:
                hands_near_head_start_time = None
        else:
            hands_near_head_start_time = None
    else:
        hands_near_head_start_time = None

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], "RGB")
    screen.blit(frame_surface, (0, 0))

    if drawing_points['left']:
        for i in range(1, len(drawing_points['left'])):
            if drawing_points['left'][i] is not None and drawing_points['left'][i-1] is not None:
                pygame.draw.line(screen, GREEN, drawing_points['left'][i-1], drawing_points['left'][i], 5)

    if drawing_points['right']:
        for i in range(1, len(drawing_points['right'])):
            if drawing_points['right'][i] is not None and drawing_points['right'][i-1] is not None:
                pygame.draw.line(screen, RED, drawing_points['right'][i-1], drawing_points['right'][i], 5)

    mode_text_left = f"Left: {'Drawing' if drawing_mode_left else 'Not Drawing'}"
    mode_text_right = f"Right: {'Drawing' if drawing_mode_right else 'Not Drawing'}"
    
    status_text_left = font.render(mode_text_left, True, BLACK)
    status_text_right = font.render(mode_text_right, True, BLACK)

    screen.blit(status_text_left, (10, 10))
    screen.blit(status_text_right, (10, 40))

    pygame.display.flip()

pygame.quit()
cv.destroyAllWindows()
cam.release()
