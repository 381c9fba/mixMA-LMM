from fastapi import FastAPI, UploadFile, File
import uvicorn

from lavis.models import load_model_and_preprocess

app = FastAPI()

# Mode loading
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_vicuna_instruct_malmm", model_type="vicuna7b", is_eval=True, device=device, memory_bank_length=10, num_frames=20,
# )



@app.post("/video")
async def upload_video(file: UploadFile = File(...)):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(await file.read())


    return {"message": "Video uploaded successfully"}


@click.command()
@click.option('--model-name', type=str, required=True, help='Name of the model to load')
@click.option('--model-name', type=str, required=True, help='Name of the model to load')
def main(model_name):
    uvicorn.run(app, host="0.0.0.0", port=8000)
    model = load_model(model_name)
    # Use the model for predictions or further processing
    predictions = model.predict(input_data)

if __name__ == '__main__':
    main()

