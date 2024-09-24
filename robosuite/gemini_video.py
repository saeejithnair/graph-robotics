import cv2
import base64
import google.generativeai as genai
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure the Gemini API
genai.configure(api_key='AIzaSyCR_B_nW2nvIzhDhZ8RE_9nVoXnrc-qoFU')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

async def process_frame(frame):
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Generate content using Gemini
    response = await model.generate_content_async([
        "Describe what's happening in this image. Be brief and focus on the main action.",
        {"mime_type": "image/jpeg", "data": img_base64}
    ])
    return response.text

async def main():
    # Open the video file
    video = cv2.VideoCapture('/Users/snair/work/scene_rep/graph-robotics/robosuite/demo_video.mov')
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Process every 1 second

    frame_count = 0
    
    # Create a ThreadPoolExecutor for running CV2 operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(executor, video.read)
            
            if not success[0]:
                break

            frame = success[1]
            frame_count += 1

            if frame_count % frame_interval == 0:
                description = await process_frame(frame)
                print(f"Frame {frame_count}: {description}")

    video.release()

if __name__ == "__main__":
    asyncio.run(main())