import io
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
import numpy as np
import base64
import model as m

app = FastAPI()

@app.post("/image")
async def process(data:dict = Body(...)):
    try:
        image_data = data["image"]
        ksize = data['ksize']
        sigmaval = data['sigmaval']
        decoded_image = base64.b64decode(image_data.split(",")[1])
       
        with open("saved.jpg","wb") as fi:
            fi.write(decoded_image)

        rects = m.get_rects("saved.jpg")

        if len(rects) ==0:
            return {'error':"There are no faces detected"}

        pil_img = Image.open("saved.jpg").convert('RGB')
        temp = np.asarray("saved.jpg")
        landmarks = m.get_landmarks(temp, rects)
        routes = m.get_faceline(landmarks)
        output = m.blur_img(routes,temp,ksize,sigmaval)
        
        # Save the output image
        output.save("output.jpg")

        # Delete the input image
        os.remove("saved.jpg")

        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process image")

@app.get("/get_output_image")
async def get_output_image():
    # Check if the output image exists
    if not os.path.exists("output.jpg"):
        return {"error": "Output image not found"}

    # Send the output image as a response that can be downloaded
    return FileResponse("output.jpg", media_type="image/jpeg", filename="output.jpg",
                        headers={"Content-Disposition": "attachment;filename=output.jpg"})

    # Delete the output image
    os.remove("output.jpg")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)