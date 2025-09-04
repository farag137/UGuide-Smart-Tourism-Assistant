import joblib
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image  # for image processing (optional)

app = FastAPI()

labelled= ['100', 'Among', 'Angry', 'Ankh', 'Aroura', 'At', 'Bad_Thinking',
       'Bandage', 'Bee', 'Belongs', 'Birth', 'Board_Game', 'Book', 'Boy',
       'Branch', 'Bread', 'Brewer', 'Builder', 'Bury', 'Canal',
       'Cloth_on_Pole', 'Cobra', 'Composite_Bow', 'Cooked', 'Corpse',
       'Dessert', 'Divide', 'Duck', 'Elephant', 'Enclosed_Mound', 'Eye',
       'Fabric', 'Face', 'Falcon', 'Fingre', 'Fish', 'Flail',
       'Folded_Cloth', 'Foot', 'Galena', 'Giraffe', 'He', 'Her', 'Hit',
       'Horn', 'King', 'Leg', 'Length_Of_a_Human_Arm', 'Life_Spirit',
       'Limit', 'Lion', 'Lizard', 'Loaf', 'Loaf_Of_Bread', 'Man',
       'Mascot', 'Meet', 'Mother', 'Mouth', 'Musical_Instrument',
       'Nile_Fish', 'Not', 'Now', 'Nurse', 'Nursing', 'Occur', 'One',
       'Owl', 'Pair', 'Papyrus_Scroll', 'Pool', 'QuailChick', 'Reed',
       'Ring', 'Rope', 'Ruler', 'Sail', 'Sandal', 'Semen', 'Small_Ring',
       'Snake', 'Soldier', 'Star', 'Stick', 'Swallow', 'This',
       'To_Be_Dead', 'To_Protect', 'To_Say', 'Turtle', 'Viper', 'Wall',
       'Water', 'Woman', 'You']


@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    """
    Upload an image file.
    """
    # Validate file type (optional)
    if not image.content_type.startswith('image/'):
        return {"error": "Invalid file type. Only images allowed."}

    # Read image data
    model = joblib.load('C:\\Users\\DELL\\Desktop\\pp\\pp\\ann_model.pkl')
    content = await image.read()
    img = Image.open(io.BytesIO(content))
    img = img.resize((250, 250))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array,verbose=2)[0]
    output = np.argmax(pred)
    output=labelled[output]
    return {"prediction": output}
    # Save image (modify path and logic as needed)
#    with open(f"/home/hady/projects/reactive_Qt/reactive_qt/uploads/{image.filename}", "wb") as f:
#       f.write(content)

    # Process image (optional)
    # img = Image.open(f"uploads/{image.filename}")
    # # Do something with the image (resize, etc.)
    # img.save(f"uploads/processed_{image.filename}")

    return {"message": f"Image '{image.filename}' uploaded successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)